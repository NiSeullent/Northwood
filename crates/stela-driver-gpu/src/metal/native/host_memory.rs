// SPDX-License-Identifier: GPL-2.0-or-later
//! Host allocations borrowed by the CPU Vulkan device.

use super::{failure, objects, raw, software::HOST_ALLOCATION};
use crate::metal::{
    error::{Error, Result},
    vulkan::{
        MemoryProperties,
        object::{BufferInfo, MemoryRequest},
        physical::Memory,
    },
};
use alloc::{boxed::Box, collections::BTreeMap};
use ash::{ext, vk};
use core::{any::Any, num::NonZeroU64, ptr::NonNull};
use stela_contracts::gate::{MemoryRange, window::IMPORT_TOKEN_WRITABLE};

#[cfg(test)]
mod tests;

/// A contiguous host allocation and the lease keeping its original mapping live.
pub struct HostAllocation {
    pub(super) pointer: NonNull<u8>,
    pub(super) bytes: NonZeroU64,
    _lease: Box<dyn Any + Send>,
}

impl HostAllocation {
    /// Gives the Vulkan backend the platform's lease on a host allocation.
    ///
    /// # Safety
    ///
    /// `pointer` must identify `bytes` of host memory that remain writable at
    /// that address until the lease is dropped. The platform must synchronize
    /// host and Vulkan access and avoid host aliases of the allocation while
    /// Vulkan holds it. The lease can be dropped on a different processor.
    pub unsafe fn new(pointer: NonNull<u8>, bytes: NonZeroU64, lease: Box<dyn Any + Send>) -> Self {
        Self {
            pointer,
            bytes,
            _lease: lease,
        }
    }
}

/// Resolves wired guest ranges to a single existing host mapping.
pub trait HostMemoryExporter: Send + Sync {
    /// The mapping must cover the supplied ranges in order, for their exact
    /// combined length. Fragmented mappings may be refused.
    fn lease(&self, ranges: &[MemoryRange]) -> Result<HostAllocation>;
}

pub(super) struct Import {
    memory: vk::DeviceMemory,
    pub(super) allocation: HostAllocation,
    memory_type: u32,
    pub(super) token: NonZeroU64,
}

impl Import {
    pub(super) fn check_binding(&self, needs: vk::MemoryRequirements, offset: u64) -> Result<()> {
        if needs.alignment == 0 || !offset.is_multiple_of(needs.alignment) {
            return Err(Error::Invalid("host allocation buffer offset"));
        }
        if offset
            .checked_add(needs.size)
            .is_none_or(|end| end > self.allocation.bytes.get())
        {
            return Err(Error::Invalid("host allocation buffer range"));
        }
        if needs.memory_type_bits & (1 << self.memory_type) == 0 {
            return Err(Error::Unsupported("host allocation buffer memory type"));
        }
        Ok(())
    }

    /// Vulkan no longer uses this allocation. Release Vulkan's handle before
    /// the platform can remove the mapping.
    pub(super) fn free(self, device: &ash::Device) {
        // SAFETY: only the backend that owns the device and observes its
        // submissions complete calls this method.
        unsafe { device.free_memory(self.memory, None) };
    }
}

pub(super) struct Imports {
    exporter: Box<dyn HostMemoryExporter>,
    functions: ext::external_memory_host::Device,
    physical: vk::PhysicalDevice,
    alignment: NonZeroU64,
    pub(super) owned: BTreeMap<u64, Import>,
}

impl Imports {
    pub(super) fn new(
        exporter: Box<dyn HostMemoryExporter>,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        alignment: NonZeroU64,
    ) -> Self {
        Self {
            exporter,
            functions: ext::external_memory_host::Device::new(instance, device),
            physical,
            alignment,
            owned: BTreeMap::new(),
        }
    }

    pub(super) fn admit_buffer(&self, instance: &ash::Instance, buffer: &BufferInfo) -> Result<()> {
        let mut usage = vk::BufferUsageFlags2CreateInfoKHR::default()
            .usage(vk::BufferUsageFlags2KHR::from_raw(buffer.usage.bits()));
        let info = vk::PhysicalDeviceExternalBufferInfo::default()
            .handle_type(HOST_ALLOCATION)
            .flags(vk::BufferCreateFlags::from_raw(buffer.flags.bits()))
            .push_next(&mut usage);
        let mut properties = vk::ExternalBufferProperties::default();
        // SAFETY: the physical device belongs to this instance and the
        // chained query structures outlive the call.
        unsafe {
            instance.get_physical_device_external_buffer_properties(
                self.physical,
                &info,
                &mut properties,
            )
        };
        if !super::software::host_importable(properties.external_memory_properties) {
            return Err(Error::Unsupported("host allocation for this Vulkan buffer"));
        }
        Ok(())
    }

    pub(super) fn allocate(
        &self,
        device: &ash::Device,
        memory: &Memory,
        ranges: &[MemoryRange],
        token: NonZeroU64,
    ) -> Result<(vk::DeviceMemory, Import)> {
        let bits = token.get();
        if bits & IMPORT_TOKEN_WRITABLE == 0 || bits & !IMPORT_TOKEN_WRITABLE == 0 {
            return Err(Error::Unsupported(
                "host allocation requires a bidirectional guest wire",
            ));
        }
        let bytes = range_bytes(ranges)?;
        let allocation = self.exporter.lease(ranges)?;
        check_allocation(&allocation, bytes, self.alignment)?;
        let mut properties = vk::MemoryHostPointerPropertiesEXT::default();
        // SAFETY: the lease owns the checked host address for the duration
        // of the query and any successful import.
        unsafe {
            (self.functions.fp().get_memory_host_pointer_properties_ext)(
                self.functions.device(),
                HOST_ALLOCATION,
                allocation.pointer.as_ptr().cast(),
                &mut properties,
            )
        }
        .result()
        .map_err(failure)?;
        let request = MemoryRequest {
            required: MemoryProperties::HOST_VISIBLE | MemoryProperties::HOST_COHERENT,
            preferred: MemoryProperties::EMPTY,
            mapped: false,
        };
        let memory_type = objects::memory_candidates(memory, properties.memory_type_bits, request)
            .first()
            .copied()
            .ok_or(Error::Unsupported("coherent Vulkan host allocation memory"))?;
        let mut source = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(HOST_ALLOCATION)
            .host_pointer(allocation.pointer.as_ptr().cast());
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(bytes.get())
            .memory_type_index(memory_type)
            .push_next(&mut source)
            .push_next(&mut flags);
        // SAFETY: the create info outlives the call and the successful import
        // retains the mapping's lease.
        let memory = unsafe { device.allocate_memory(&info, None) }.map_err(failure)?;
        Ok((
            memory,
            Import {
                memory,
                allocation,
                memory_type,
                token,
            },
        ))
    }

    pub(super) fn remove(&mut self, memory: vk::DeviceMemory) -> Option<Import> {
        self.owned.remove(&raw(memory))
    }

    pub(super) fn release(&mut self, device: &ash::Device) {
        for allocation in core::mem::take(&mut self.owned).into_values() {
            allocation.free(device);
        }
    }
}

fn range_bytes(ranges: &[MemoryRange]) -> Result<NonZeroU64> {
    ranges
        .iter()
        .try_fold(0u64, |sum, range| {
            if range.bytes == 0 || range.address.checked_add(range.bytes).is_none() {
                return None;
            }
            sum.checked_add(range.bytes)
        })
        .and_then(NonZeroU64::new)
        .ok_or(Error::Invalid("host allocation guest ranges"))
}

fn check_allocation(
    allocation: &HostAllocation,
    bytes: NonZeroU64,
    alignment: NonZeroU64,
) -> Result<()> {
    if allocation.bytes != bytes {
        return Err(Error::Invalid("host allocation length"));
    }
    let address = allocation.pointer.as_ptr().addr() as u64;
    let alignment = alignment.get();
    if address.checked_add(bytes.get()).is_none()
        || !address.is_multiple_of(alignment)
        || !bytes.get().is_multiple_of(alignment)
    {
        return Err(Error::Invalid("Vulkan host allocation alignment"));
    }
    Ok(())
}
