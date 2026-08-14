//! Exact compatibility between one guest plane and one imported linear image.
//!
//! Device-level format support is only the outer gate. A guest-backed image is
//! correct only when the driver's actual subresource layout places its first
//! texel at the plane's declared offset, uses the same row pitch, fits inside
//! the retained packed alias, and accepts a memory type the host pointer also
//! accepts. Those facts belong together because every one participates in the
//! same `vkBindImageMemory` equation.

use ash::vk;

use super::context::DeviceContext;
use super::types::GuestTargetBacking;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WindowPlan {
    bind_offset: u64,
    memory_type_index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WindowRefusal {
    UnsupportedTopology,
    HostImportUnavailable,
    HostPointerMisaligned,
    SubresourceAfterPlane,
    BindOffsetMisaligned,
    RowPitchMismatch,
    AllocationTooShort,
    NoMemoryType,
    DedicatedBindingRequired,
    CreateImage(vk::Result),
    PointerProperties(vk::Result),
    AllocateMemory(vk::Result),
    BindImage(vk::Result),
}

impl WindowRefusal {
    pub(super) fn slug(self) -> &'static str {
        match self {
            Self::UnsupportedTopology => "discrete_topology",
            Self::HostImportUnavailable => "no_host_import",
            Self::HostPointerMisaligned => "host_pointer_misaligned",
            Self::SubresourceAfterPlane => "subresource_after_plane",
            Self::BindOffsetMisaligned => "bind_offset_misaligned",
            Self::RowPitchMismatch => "row_pitch_mismatch",
            Self::AllocationTooShort => "allocation_too_short",
            Self::NoMemoryType => "no_memory_type",
            Self::DedicatedBindingRequired => "dedicated_binding_required",
            Self::CreateImage(_) => "create_failed",
            Self::PointerProperties(_) => "pointer_properties_failed",
            Self::AllocateMemory(_) => "allocate_failed",
            Self::BindImage(_) => "bind_failed",
        }
    }

    pub(super) fn result(self) -> Option<vk::Result> {
        match self {
            Self::CreateImage(result)
            | Self::PointerProperties(result)
            | Self::AllocateMemory(result)
            | Self::BindImage(result) => Some(result),
            _ => None,
        }
    }
}

impl crate::observe::Decline for WindowRefusal {
    fn slug(&self) -> &'static str {
        (*self).slug()
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        self.result()
            .map(|result| vec![("result", format!("{result:?}"))])
            .unwrap_or_default()
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "the planner checks one independently reported value per term of the image-binding equation"
)]
fn plan_window(
    layout: vk::SubresourceLayout,
    requirements: vk::MemoryRequirements,
    allocation_len: u64,
    plane_offset: u64,
    guest_row_pitch: u64,
    pointer_memory_type_bits: u32,
    memory_type_index: Option<u32>,
    requires_dedicated: bool,
) -> Result<WindowPlan, WindowRefusal> {
    let bind_offset = plane_offset
        .checked_sub(layout.offset)
        .ok_or(WindowRefusal::SubresourceAfterPlane)?;
    if requirements.alignment == 0 || !bind_offset.is_multiple_of(requirements.alignment) {
        return Err(WindowRefusal::BindOffsetMisaligned);
    }
    if layout.row_pitch != guest_row_pitch {
        return Err(WindowRefusal::RowPitchMismatch);
    }
    let required_end = bind_offset
        .checked_add(requirements.size)
        .ok_or(WindowRefusal::AllocationTooShort)?;
    if required_end > allocation_len {
        return Err(WindowRefusal::AllocationTooShort);
    }
    if requirements.memory_type_bits & pointer_memory_type_bits == 0 {
        return Err(WindowRefusal::NoMemoryType);
    }
    let memory_type_index = memory_type_index.ok_or(WindowRefusal::NoMemoryType)?;
    if requires_dedicated && bind_offset != 0 {
        return Err(WindowRefusal::DedicatedBindingRequired);
    }
    Ok(WindowPlan {
        bind_offset,
        memory_type_index,
    })
}

pub(super) struct ImportedTarget {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
}

/// Create a linear image whose storage is the guest surface allocation itself.
///
/// A refusal is an optional-rail answer: callers keep the ordinary optimal
/// resident. Once this returns an image, its memory is a one-to-one import and
/// must be freed directly rather than entering the resident recycle slab.
pub(super) unsafe fn create(
    ctx: &DeviceContext,
    backing: GuestTargetBacking,
    width: u32,
    height: u32,
    format: vk::Format,
    mut usage: vk::ImageUsageFlags,
) -> Result<ImportedTarget, WindowRefusal> {
    use crate::backend::vulkan::caps::memory_topology::MemoryTopology;

    if ctx.caps.memory.topology != MemoryTopology::Unified {
        return Err(WindowRefusal::UnsupportedTopology);
    }
    let ext = ctx
        .external_memory_host
        .as_ref()
        .ok_or(WindowRefusal::HostImportUnavailable)?;
    let alignment = ctx.caps.host_pointer.min_alignment;
    if alignment == 0
        || !(backing.allocation_host_ptr as u64).is_multiple_of(alignment)
        || !backing.allocation_len.is_multiple_of(alignment)
    {
        return Err(WindowRefusal::HostPointerMisaligned);
    }
    if ctx.features.attachment_feedback_loop_layout {
        usage |= vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
    }
    let handle = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
    let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle);
    let create = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::LINEAR)
        .usage(usage)
        .initial_layout(vk::ImageLayout::PREINITIALIZED)
        .push_next(&mut external);
    let image =
        unsafe { ctx.device.create_image(&create, None) }.map_err(WindowRefusal::CreateImage)?;

    let result = (|| {
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        let info = vk::ImageMemoryRequirementsInfo2::default().image(image);
        unsafe {
            ctx.device
                .get_image_memory_requirements2(&info, &mut requirements)
        };
        let layout = unsafe {
            ctx.device.get_image_subresource_layout(
                image,
                vk::ImageSubresource {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    array_layer: 0,
                },
            )
        };
        let mut pointer = vk::MemoryHostPointerPropertiesEXT::default();
        let pointer_result = unsafe {
            (ext.fp().get_memory_host_pointer_properties_ext)(
                ext.device(),
                handle,
                backing.allocation_host_ptr as *const std::ffi::c_void,
                &mut pointer,
            )
        };
        if pointer_result != vk::Result::SUCCESS {
            return Err(WindowRefusal::PointerProperties(pointer_result));
        }
        let compatible =
            pointer.memory_type_bits & requirements.memory_requirements.memory_type_bits;
        let picked = ctx.memory_type_with(
            compatible,
            backing.allocation_len,
            &ctx.caps
                .memory_request(crate::backend::vulkan::caps::MemoryClass::Upload),
        );
        let plan = plan_window(
            layout,
            requirements.memory_requirements,
            backing.allocation_len,
            backing.plane_offset,
            backing.row_pitch,
            pointer.memory_type_bits,
            picked.map(|pick| pick.index),
            dedicated.requires_dedicated_allocation != 0,
        )?;
        let mut host_import = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(handle)
            .host_pointer(backing.allocation_host_ptr as *mut std::ffi::c_void);
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(backing.allocation_len)
            .memory_type_index(plan.memory_type_index)
            .push_next(&mut host_import);
        let memory = unsafe { ctx.device.allocate_memory(&allocate, None) }
            .map_err(WindowRefusal::AllocateMemory)?;
        if let Err(result) = unsafe {
            ctx.device
                .bind_image_memory(image, memory, plan.bind_offset)
        } {
            unsafe { ctx.device.free_memory(memory, None) };
            return Err(WindowRefusal::BindImage(result));
        }
        Ok(ImportedTarget { image, memory })
    })();
    if result.is_err() {
        unsafe { ctx.device.destroy_image(image, None) };
    }
    result
}

/// Probe the complete binding equation for one live guest surface.
///
/// This creates no memory and changes no rendering behavior. It asks the
/// driver for the linear image's actual layout and memory requirements, then
/// checks them against the guest allocation. The behavior implementation will
/// consume this same planner rather than reconstructing the admission rule.
///
/// # Safety
///
/// `host_ptr..host_ptr + allocation_len` must remain a live host mapping while
/// the pointer-properties query runs, and `ctx` must own the logical device.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn probe_window(
    ctx: &DeviceContext,
    host_ptr: usize,
    allocation_len: u64,
    plane_offset: u64,
    guest_row_pitch: u64,
    width: u32,
    height: u32,
    format: vk::Format,
    mut usage: vk::ImageUsageFlags,
) {
    use crate::backend::vulkan::caps::memory_topology::MemoryTopology;

    if ctx.caps.memory.topology != MemoryTopology::Unified {
        crate::observe::off(format!(
            "vk_linear_target_window verdict=discrete_topology format={format:?} {width}x{height}"
        ));
        return;
    }
    let Some(ext) = ctx.external_memory_host.as_ref() else {
        crate::observe::off(format!(
            "vk_linear_target_window verdict=no_host_import format={format:?} {width}x{height}"
        ));
        return;
    };
    if ctx.features.attachment_feedback_loop_layout {
        usage |= vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
    }
    let handle = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
    let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle);
    let create = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::LINEAR)
        .usage(usage)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut external);
    let image = match unsafe { ctx.device.create_image(&create, None) } {
        Ok(image) => image,
        Err(result) => {
            crate::observe::off(format!(
                "vk_linear_target_window verdict=create_failed result={result:?} format={format:?} {width}x{height}"
            ));
            return;
        }
    };
    let requirements = unsafe { ctx.device.get_image_memory_requirements(image) };
    let layout = unsafe {
        ctx.device.get_image_subresource_layout(
            image,
            vk::ImageSubresource {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                array_layer: 0,
            },
        )
    };
    let mut pointer = vk::MemoryHostPointerPropertiesEXT::default();
    let pointer_result = unsafe {
        (ext.fp().get_memory_host_pointer_properties_ext)(
            ext.device(),
            handle,
            host_ptr as *const std::ffi::c_void,
            &mut pointer,
        )
    };
    let pointer_bits = if pointer_result == vk::Result::SUCCESS {
        pointer.memory_type_bits
    } else {
        0
    };
    let compatible_bits = pointer_bits & requirements.memory_type_bits;
    let picked = ctx.memory_type_with(
        compatible_bits,
        allocation_len,
        &ctx.caps
            .memory_request(crate::backend::vulkan::caps::MemoryClass::Upload),
    );
    let plan = plan_window(
        layout,
        requirements,
        allocation_len,
        plane_offset,
        guest_row_pitch,
        pointer_bits,
        picked.map(|pick| pick.index),
        false,
    );
    let verdict = match (plan, picked) {
        (Ok(_), Some(_)) => "alias_exact",
        (Ok(_), None) => WindowRefusal::NoMemoryType.slug(),
        (Err(reason), _) => reason.slug(),
    };
    let bind_offset = plan.ok().map(|p| p.bind_offset).unwrap_or(u64::MAX);
    crate::observe::off(format!(
        "vk_linear_target_window verdict={verdict} format={format:?} {width}x{height} allocation_len={allocation_len} plane_offset={plane_offset} guest_row_pitch={guest_row_pitch} layout_offset={} layout_row_pitch={} requirements_size={} requirements_align={} bind_offset={bind_offset} image_type_bits=0x{:x} pointer_type_bits=0x{pointer_bits:x} compatible_type_bits=0x{compatible_bits:x} memory_type={}",
        layout.offset,
        layout.row_pitch,
        requirements.size,
        requirements.alignment,
        requirements.memory_type_bits,
        picked.map(|p| p.index.to_string()).unwrap_or_else(|| "none".into()),
    ));
    unsafe { ctx.device.destroy_image(image, None) };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(offset: u64, row_pitch: u64) -> vk::SubresourceLayout {
        vk::SubresourceLayout {
            offset,
            size: 0,
            row_pitch,
            array_pitch: 0,
            depth_pitch: 0,
        }
    }

    fn requirements(size: u64, alignment: u64, bits: u32) -> vk::MemoryRequirements {
        vk::MemoryRequirements {
            size,
            alignment,
            memory_type_bits: bits,
        }
    }

    #[test]
    fn an_exact_window_derives_the_binding_offset() {
        assert_eq!(
            plan_window(
                layout(0, 7680),
                requirements(8 << 20, 4096, 0b110),
                12 << 20,
                4096,
                7680,
                0b010,
                Some(1),
                false,
            ),
            Ok(WindowPlan {
                bind_offset: 4096,
                memory_type_index: 1,
            })
        );
    }

    #[test]
    fn every_part_of_the_binding_equation_can_refuse() {
        let req = requirements(8192, 4096, 0b010);
        assert_eq!(
            plan_window(
                layout(8192, 256),
                req,
                16384,
                4096,
                256,
                0b010,
                Some(1),
                false
            ),
            Err(WindowRefusal::SubresourceAfterPlane)
        );
        assert_eq!(
            plan_window(layout(0, 256), req, 16384, 2048, 256, 0b010, Some(1), false),
            Err(WindowRefusal::BindOffsetMisaligned)
        );
        assert_eq!(
            plan_window(layout(0, 512), req, 16384, 4096, 256, 0b010, Some(1), false),
            Err(WindowRefusal::RowPitchMismatch)
        );
        assert_eq!(
            plan_window(layout(0, 256), req, 8192, 4096, 256, 0b010, Some(1), false),
            Err(WindowRefusal::AllocationTooShort)
        );
        assert_eq!(
            plan_window(layout(0, 256), req, 16384, 4096, 256, 0b100, None, false),
            Err(WindowRefusal::NoMemoryType)
        );
        assert_eq!(
            plan_window(
                layout(0, 256),
                req,
                16384,
                4096,
                256,
                0b010,
                Some(1),
                true,
            ),
            Err(WindowRefusal::DedicatedBindingRequired)
        );
    }
}
