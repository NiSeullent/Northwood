//! Owned HPET timing for the standalone USB diagnostics.
use crate::paging::{ActivatedTables, Tables, config};
use alloc::{boxed::Box, rc::Rc};
use stela_contracts::refuse;
use stela_drivers::bus::pci::power::Timer;

pub(crate) struct Prepared(config::Prepared);

impl Prepared {
    pub(crate) fn capture() -> crate::Result<Self> {
        let mut source = config::Prepared::capture()
            .map_err(|error| refuse!("USB clock resource capture: {error}", %error))?;
        source
            .capture_hpet()
            .map_err(|error| refuse!("USB HPET table capture: {error}", %error))?;
        Ok(Self(source))
    }

    pub(crate) fn map(&mut self, tables: &mut Tables) -> crate::Result<()> {
        self.0
            .map(tables)
            .map_err(|error| refuse!("USB clock aperture: {error}", %error))
    }

    /// # Safety
    /// Same exclusive standalone BSP/root and complete USB/HPET handoff
    /// through shutdown. Firmware, SMM, and other actors cannot access or
    /// change those resources after EBS.
    pub(crate) unsafe fn attach(self, root: &ActivatedTables) -> crate::Result<Rc<dyn Timer>> {
        // SAFETY: caller keeps the original standalone root and firmware handoff.
        let source = unsafe { config::Source::activate(root, self.0) }
            .map_err(|error| refuse!("USB clock root: {error}", %error))?;
        let source = Box::leak(Box::new(source));
        // SAFETY: caller grants the complete original HPET page until shutdown.
        let owner = unsafe { source.take_hpet() }
            .map_err(|error| refuse!("USB HPET handoff: {error}", %error))?;
        let prepared = crate::native_clock::Prepared::standalone_hpet(owner)
            .map_err(|error| refuse!("USB HPET counter: {error}", ?error))?;
        // Keep the controller through the final disk checks and shutdown.
        // USB may retain a timer with its queues on an uncertain device exit.
        Box::leak(Box::new(prepared))
            .pci_minimum()
            .map_err(|error| refuse!("USB HPET minimum: {error}", ?error))
    }
}
