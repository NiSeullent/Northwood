// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;

impl Namespace {
    /// Keep this namespace private, before publishing it through any other
    /// block route. The owned attachment must cover the controller and all
    /// of its aliases, including access from firmware and other hosts.
    /// Return the original route on refusal so its outstanding work and
    /// resources still have an owner.
    #[expect(
        clippy::result_large_err,
        reason = "the error returns the original namespace and its owners without allocating"
    )]
    pub fn into_exclusive(mut self) -> Result<ExclusiveNamespace, (Self, BlockError)> {
        if let Err(error) = self.claim_exclusive() {
            return Err((self, error));
        }
        self.identity.features.discard = None;
        self.identity.features.write_zeroes = None;
        Ok(ExclusiveNamespace {
            namespace: self,
            refused: None,
        })
    }

    fn claim_exclusive(&mut self) -> Result<(), BlockError> {
        if Rc::get_mut(&mut self.controller).is_none() {
            return Err(BlockError::Unsupported("NVMe controller has another owner"));
        }
        if let Some(error) = &self.removed {
            return Err(error.clone());
        }
        let controller = self.controller.clone();
        let mut controller = controller.borrow_mut();
        if controller.owned.is_none() {
            return Err(BlockError::Unsupported("NVMe controller has no DMA owner"));
        }
        let scans = self.scanned;
        check_admission(&mut controller, scans)?;
        // Ordinary rescans may retain the old Identity on a refused
        // Identify. Establish the claimed identity from a fresh answer.
        self.refresh_namespace(&mut controller, true)?;
        check_admission(&mut controller, scans)
    }
}

fn check_admission(controller: &mut Controller, scans: u64) -> Result<(), BlockError> {
    controller.ready()?;
    if controller.recovery.is_some()
        || controller.tags.iter().any(|tag| !matches!(tag, Tag::Free))
        || controller.held.iter().any(|pages| !pages.is_empty())
        || !controller.aborts.is_empty()
    {
        return Err(BlockError::Protocol(
            "NVMe controller still owns outstanding work",
        ));
    }
    match controller.registers.read32(STATUS)? {
        REMOVED => return Err(removed()),
        status if status & (READY | FATAL | PROCESSING_PAUSED) != READY => {
            return Err(BlockError::Device { status });
        }
        _ => {}
    }
    if controller.events.request.is_some() {
        controller.admin_completion(None)?;
    }
    if controller.scans != scans
        || controller.events.completed.is_some()
        || (controller.events.supported != 0 && controller.events.request.is_none())
    {
        return Err(BlockError::Unsupported(
            "exclusive NVMe namespace epoch changed",
        ));
    }
    Ok(())
}

impl ExclusiveNamespace {
    fn run(&mut self, command: Command) -> Finished {
        if let Some(error) = &self.refused {
            return Err(error.clone());
        }
        let identity = &self.namespace.identity;
        let bytes = match &command {
            Command::Read { bytes, .. } => Some(*bytes),
            Command::Write { data, .. } => Some(data.len()),
            _ => None,
        };
        if bytes.is_some_and(|bytes| bytes == 0 || bytes > identity.features.max_transfer) {
            return Err(BlockError::Unsupported(
                "invalid exclusive NVMe transfer size",
            ));
        }
        let command = command.within(identity, 0)?;
        let mut controller = self.namespace.controller.borrow_mut();
        let scans = self.namespace.scanned;
        let mut validate = |controller: &mut Controller| {
            check_admission(controller, scans).inspect_err(|error| {
                self.refused = Some(error.clone());
            })
        };
        validate(&mut controller)?;
        // Admit under the same controller borrow as the epoch check, with
        // the retry budget exhausted. An error CQE cannot resubmit it and
        // reset recovery cannot replay it. An unproven end leaves its pages
        // in held and the owned controller retains their mappings.
        let tag = controller.start_with_retries(
            self.namespace.namespace,
            identity.block_size as usize,
            command,
            RETRIES,
        )?;
        let result = controller.wait_for_tag(tag as usize, None)?;
        // A notice may arrive while this command is running. The command
        // may already have taken effect, but do not publish its result.
        validate(&mut controller)?;
        Ok(result)
    }
}

impl BlockDevice for ExclusiveNamespace {
    fn identity(&self) -> &Identity {
        &self.namespace.identity
    }

    fn read(&mut self, lba: u64, data: &mut [u8]) -> Result<(), BlockError> {
        let result = self.run(Command::Read {
            lba,
            bytes: data.len(),
        })?;
        let bytes =
            result
                .filter(|bytes| bytes.len() == data.len())
                .ok_or(BlockError::Protocol(
                    "exclusive NVMe read returned the wrong length",
                ))?;
        data.copy_from_slice(&bytes);
        Ok(())
    }

    fn write(&mut self, lba: u64, data: &[u8]) -> Result<(), BlockError> {
        self.run(Command::Write {
            lba,
            data: data.to_vec(),
            fua: false,
        })
        .map(drop)
    }

    fn write_fua(&mut self, lba: u64, data: &[u8]) -> Result<(), BlockError> {
        self.run(Command::Write {
            lba,
            data: data.to_vec(),
            fua: true,
        })
        .map(drop)
    }

    fn flush_cache(&mut self) -> Result<(), BlockError> {
        self.run(Command::Flush).map(drop)
    }

    fn execute(&mut self, command: Command) -> Finished {
        match command {
            Command::Read { .. } | Command::Write { .. } | Command::Flush => self.run(command),
            _ => Err(BlockError::Unsupported(
                "exclusive NVMe supports read, write and flush",
            )),
        }
    }
}
