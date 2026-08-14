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
enum LayoutMode {
    DriverLinear,
    ExplicitLinear,
}

// The linear DRM modifier is the API value zero. Unlike vendor modifiers, it
// describes ordinary row-major storage and therefore lets the guest's declared
// byte offset and row pitch be stated directly in the image-create contract.
const DRM_FORMAT_MOD_LINEAR: u64 = 0;

fn subresource_aspect(mode: LayoutMode) -> vk::ImageAspectFlags {
    match mode {
        LayoutMode::DriverLinear => vk::ImageAspectFlags::COLOR,
        LayoutMode::ExplicitLinear => vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
    }
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
    ModifierQuery(vk::Result),
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
            Self::ModifierQuery(_) => "modifier_query_failed",
            Self::CreateImage(_) => "create_failed",
            Self::PointerProperties(_) => "pointer_properties_failed",
            Self::AllocateMemory(_) => "allocate_failed",
            Self::BindImage(_) => "bind_failed",
        }
    }

    pub(super) fn result(self) -> Option<vk::Result> {
        match self {
            Self::CreateImage(result)
            | Self::ModifierQuery(result)
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
    if requires_dedicated {
        return Err(WindowRefusal::DedicatedBindingRequired);
    }
    Ok(WindowPlan {
        bind_offset,
        memory_type_index,
    })
}

#[allow(
    clippy::too_many_arguments,
    reason = "the explicit layout validates every term returned for the imported image"
)]
fn plan_explicit_window(
    layout: vk::SubresourceLayout,
    requirements: vk::MemoryRequirements,
    allocation_len: u64,
    plane_offset: u64,
    guest_row_pitch: u64,
    pointer_memory_type_bits: u32,
    memory_type_index: Option<u32>,
    requires_dedicated: bool,
) -> Result<WindowPlan, WindowRefusal> {
    if layout.offset != plane_offset {
        return Err(WindowRefusal::SubresourceAfterPlane);
    }
    if layout.row_pitch != guest_row_pitch {
        return Err(WindowRefusal::RowPitchMismatch);
    }
    if requirements.alignment == 0 || requirements.size > allocation_len {
        return Err(WindowRefusal::AllocationTooShort);
    }
    if requirements.memory_type_bits & pointer_memory_type_bits == 0 {
        return Err(WindowRefusal::NoMemoryType);
    }
    let memory_type_index = memory_type_index.ok_or(WindowRefusal::NoMemoryType)?;
    if requires_dedicated {
        return Err(WindowRefusal::DedicatedBindingRequired);
    }
    Ok(WindowPlan {
        bind_offset: 0,
        memory_type_index,
    })
}

fn required_modifier_features(usage: vk::ImageUsageFlags) -> vk::FormatFeatureFlags {
    let mut required = vk::FormatFeatureFlags::empty();
    if usage.contains(vk::ImageUsageFlags::SAMPLED) {
        required |= vk::FormatFeatureFlags::SAMPLED_IMAGE;
    }
    if usage
        .intersects(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT)
    {
        required |= vk::FormatFeatureFlags::COLOR_ATTACHMENT
            | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND;
    }
    if usage.contains(vk::ImageUsageFlags::TRANSFER_SRC) {
        required |= vk::FormatFeatureFlags::TRANSFER_SRC;
    }
    if usage.contains(vk::ImageUsageFlags::TRANSFER_DST) {
        required |= vk::FormatFeatureFlags::TRANSFER_DST;
    }
    required
}

fn external_import_is_shareable(features: vk::ExternalMemoryFeatureFlags) -> bool {
    features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
        && !features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY)
}

unsafe fn explicit_linear_supported(
    ctx: &DeviceContext,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<bool, WindowRefusal> {
    if !ctx.features.image_drm_format_modifier {
        return Ok(false);
    }
    let key = (format.as_raw(), usage.as_raw());
    if let Some(answer) = ctx
        .explicit_linear_support
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(&key)
        .copied()
    {
        return Ok(answer);
    }

    let answer = unsafe { query_explicit_linear_support(ctx, format, usage) }?;
    ctx.explicit_linear_support
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .insert(key, answer);
    Ok(answer)
}

unsafe fn query_explicit_linear_support(
    ctx: &DeviceContext,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<bool, WindowRefusal> {
    let mut modifier_list = vk::DrmFormatModifierPropertiesListEXT::default();
    let mut properties = vk::FormatProperties2::default().push_next(&mut modifier_list);
    unsafe {
        ctx.instance
            .get_physical_device_format_properties2(ctx.pd, format, &mut properties)
    };
    let mut modifiers = vec![
        vk::DrmFormatModifierPropertiesEXT::default();
        modifier_list.drm_format_modifier_count as usize
    ];
    let mut modifier_list = vk::DrmFormatModifierPropertiesListEXT::default()
        .drm_format_modifier_properties(&mut modifiers);
    let mut properties = vk::FormatProperties2::default().push_next(&mut modifier_list);
    unsafe {
        ctx.instance
            .get_physical_device_format_properties2(ctx.pd, format, &mut properties)
    };
    let required = required_modifier_features(usage);
    if !modifiers.iter().any(|modifier| {
        modifier.drm_format_modifier == DRM_FORMAT_MOD_LINEAR
            && modifier.drm_format_modifier_plane_count == 1
            && modifier
                .drm_format_modifier_tiling_features
                .contains(required)
    }) {
        return Ok(false);
    }

    let handle = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
    let mut modifier = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
        .drm_format_modifier(DRM_FORMAT_MOD_LINEAR)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let mut external = vk::PhysicalDeviceExternalImageFormatInfo::default().handle_type(handle);
    let info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(usage)
        .push_next(&mut modifier)
        .push_next(&mut external);
    let mut external_properties = vk::ExternalImageFormatProperties::default();
    let mut properties = vk::ImageFormatProperties2::default().push_next(&mut external_properties);
    unsafe {
        ctx.instance
            .get_physical_device_image_format_properties2(ctx.pd, &info, &mut properties)
    }
    .map_err(WindowRefusal::ModifierQuery)?;
    Ok(external_import_is_shareable(
        external_properties
            .external_memory_properties
            .external_memory_features,
    ))
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
    if ctx.external_memory_host.is_none() {
        return Err(WindowRefusal::HostImportUnavailable);
    }
    let alignment = ctx.caps.host_pointer.min_alignment;
    if alignment == 0
        || !(backing.allocation_host_ptr as u64).is_multiple_of(alignment)
        || !backing.allocation_len.is_multiple_of(alignment)
    {
        return Err(WindowRefusal::HostPointerMisaligned);
    }
    if ctx.features.attachment_feedback_loop_layout
        && usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
    {
        usage |= vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
    }
    if unsafe { explicit_linear_supported(ctx, format, usage) }? {
        let explicit = unsafe {
            create_with_layout(
                ctx,
                backing,
                width,
                height,
                format,
                usage,
                LayoutMode::ExplicitLinear,
            )
        };
        if explicit.is_ok() {
            return explicit;
        }
        // A format/modifier combination is structural, but an individual row
        // pitch can still violate that modifier's alignment. Preserve the
        // ordinary exact-pitch route where it happens to fit.
        let ordinary = unsafe {
            create_with_layout(
                ctx,
                backing,
                width,
                height,
                format,
                usage,
                LayoutMode::DriverLinear,
            )
        };
        return ordinary.or(explicit);
    }
    unsafe {
        create_with_layout(
            ctx,
            backing,
            width,
            height,
            format,
            usage,
            LayoutMode::DriverLinear,
        )
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn create_with_layout(
    ctx: &DeviceContext,
    backing: GuestTargetBacking,
    width: u32,
    height: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    mode: LayoutMode,
) -> Result<ImportedTarget, WindowRefusal> {
    let ext = ctx
        .external_memory_host
        .as_ref()
        .ok_or(WindowRefusal::HostImportUnavailable)?;
    let handle = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
    let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle);
    let base = vk::ImageCreateInfo::default()
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
        .usage(usage)
        .initial_layout(vk::ImageLayout::PREINITIALIZED);
    let image = match mode {
        LayoutMode::DriverLinear => {
            let create = base
                .tiling(vk::ImageTiling::LINEAR)
                .push_next(&mut external);
            unsafe { ctx.device.create_image(&create, None) }
        }
        LayoutMode::ExplicitLinear => {
            let plane_layout = [vk::SubresourceLayout {
                offset: backing.plane_offset,
                size: 0,
                row_pitch: backing.row_pitch,
                array_pitch: 0,
                depth_pitch: 0,
            }];
            let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
                .drm_format_modifier(DRM_FORMAT_MOD_LINEAR)
                .plane_layouts(&plane_layout);
            let create = base
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .push_next(&mut external)
                .push_next(&mut explicit);
            unsafe { ctx.device.create_image(&create, None) }
        }
    }
    .map_err(WindowRefusal::CreateImage)?;

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
                    // Modifier images describe memory planes, not format
                    // aspects. A single-plane linear modifier therefore has
                    // exactly MEMORY_PLANE_0_EXT; ordinary linear images keep
                    // the colour aspect query.
                    aspect_mask: subresource_aspect(mode),
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
        let plan = match mode {
            LayoutMode::DriverLinear => plan_window(
                layout,
                requirements.memory_requirements,
                backing.allocation_len,
                backing.plane_offset,
                backing.row_pitch,
                pointer.memory_type_bits,
                picked.map(|pick| pick.index),
                dedicated.requires_dedicated_allocation != 0,
            ),
            LayoutMode::ExplicitLinear => plan_explicit_window(
                layout,
                requirements.memory_requirements,
                backing.allocation_len,
                backing.plane_offset,
                backing.row_pitch,
                pointer.memory_type_bits,
                picked.map(|pick| pick.index),
                dedicated.requires_dedicated_allocation != 0,
            ),
        }?;
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
            plan_window(layout(0, 256), req, 16384, 4096, 256, 0b010, Some(1), true,),
            Err(WindowRefusal::DedicatedBindingRequired)
        );
    }

    #[test]
    fn explicit_layout_binds_the_import_at_zero() {
        assert_eq!(
            plan_explicit_window(
                layout(4096, 7040),
                requirements(8 << 20, 4096, 0b110),
                12 << 20,
                4096,
                7040,
                0b010,
                Some(1),
                false,
            ),
            Ok(WindowPlan {
                bind_offset: 0,
                memory_type_index: 1,
            })
        );
    }

    #[test]
    fn explicit_layout_must_be_returned_exactly() {
        let req = requirements(8192, 4096, 0b010);
        assert_eq!(
            plan_explicit_window(layout(0, 256), req, 16384, 4096, 256, 0b010, Some(1), false,),
            Err(WindowRefusal::SubresourceAfterPlane)
        );
        assert_eq!(
            plan_explicit_window(
                layout(4096, 512),
                req,
                16384,
                4096,
                256,
                0b010,
                Some(1),
                false,
            ),
            Err(WindowRefusal::RowPitchMismatch)
        );
    }

    #[test]
    fn modifier_features_follow_the_declared_usage() {
        let features = required_modifier_features(
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
        );
        assert!(features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE));
        assert!(features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT));
        assert!(features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND));
        assert!(features.contains(vk::FormatFeatureFlags::TRANSFER_SRC));
        assert!(features.contains(vk::FormatFeatureFlags::TRANSFER_DST));
    }

    #[test]
    fn explicit_layout_queries_the_memory_plane() {
        assert_eq!(
            subresource_aspect(LayoutMode::DriverLinear),
            vk::ImageAspectFlags::COLOR
        );
        assert_eq!(
            subresource_aspect(LayoutMode::ExplicitLinear),
            vk::ImageAspectFlags::MEMORY_PLANE_0_EXT
        );
    }

    #[test]
    fn explicit_import_requires_a_non_dedicated_importable_image() {
        assert!(external_import_is_shareable(
            vk::ExternalMemoryFeatureFlags::IMPORTABLE
        ));
        assert!(!external_import_is_shareable(
            vk::ExternalMemoryFeatureFlags::empty()
        ));
        assert!(!external_import_is_shareable(
            vk::ExternalMemoryFeatureFlags::IMPORTABLE
                | vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY
        ));
    }
}
