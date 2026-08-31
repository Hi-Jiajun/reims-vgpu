//! `CmdHeapTextureSizeAndAlign` request decode and host requirement query.
//!
//! The command carries a `PGSerializedTextureDescriptor` record. The recovered
//! Apple host routine reconstructs an `MTLTextureDescriptor` and returns the
//! device's `MTLSizeAndAlign` as two little-endian `u64`s.

use crate::contract::endian::{ld32, ld64, st64};
use reims_vgpu_wire::ops::texture as wire;

pub const REQUEST_HEADER_LEN: usize = 24;
pub const REPLY_LEN: usize = 16;
/// The embedded record is `heapTextureSizeAndAlignWithDescriptor:`, so its tag
/// is that selector's opcode and its length is that record's length. Both come
/// from the crate that derived them rather than being written again here.
pub const SERIALIZED_TEXTURE_TAG: u32 = wire::OPCODE_HEAP_TEXTURE_SIZE_AND_ALIGN;
pub const SERIALIZED_TEXTURE_LEN: usize = wire::HEAP_TEXTURE_SIZE_AND_ALIGN_TOTAL_LEN as usize;
pub const TEXTURE_BODY_LEN: usize = wire::TEXTURE_DESCRIPTOR_LEN;
/// The same descriptor once the guest's serializer has a texture-descriptor
/// capability on: eight bytes wider, with `usage` promoted out of the packed
/// word into a `u32` and a four-channel swizzle appended.
///
/// Two different flags select it depending on the record — `SwizzledTextures`
/// for the plain creation, `TextureDescriptor2` for the four that embed it —
/// and neither is a length the caller may infer. Every record carrying this
/// body has an opcode of its own; see
/// [`crate::runtime::decode::resource::decode_heap_texture`].
pub const WIDE_TEXTURE_BODY_LEN: usize = wire::WIDE_TEXTURE_DESCRIPTOR_LEN;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureDescriptor {
    pub texture_type: u8,
    pub framebuffer_only: bool,
    pub is_drawable: bool,
    pub allow_gpu_optimized_contents: bool,
    /// `MTLTextureUsage` mask. Eight bits of the packed word on the narrow
    /// body and a field of its own on the wide one, so it is held at the wider
    /// of the two — narrowing it here would drop any bit above 7 that a wide
    /// descriptor sets.
    pub usage: u32,
    pub pixel_format: u16,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mipmap_level_count: u16,
    pub sample_count: u16,
    pub array_length: u16,
    pub resource_options: u16,
    pub protection_options: u64,
    /// `MTLTextureSwizzleChannels` as four raw `MTLTextureSwizzle` ordinals in
    /// red, green, blue, alpha order — the same encoding the texture-view swizzle
    /// view carries, so [`crate::contract::pixel_format::swizzle_plan`] reads
    /// both.
    ///
    /// `None` on the narrow body, which has no such field at all. That is not
    /// the same as the identity: a reader must not turn an absent swizzle into
    /// a present one.
    pub swizzle: Option<[u8; 4]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    pub task_id: u32,
    pub reply_gva: u64,
    pub reply_len: u64,
    pub descriptor: TextureDescriptor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizeAndAlign {
    pub size: u64,
    pub align: u64,
}

impl SizeAndAlign {
    pub fn encode(self) -> [u8; REPLY_LEN] {
        let mut out = [0u8; REPLY_LEN];
        st64(&mut out[0..8], self.size);
        st64(&mut out[8..16], self.align);
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryError {
    ShortPayload,
    BadReplyLength,
    BadSerializerLength,
    UnknownSerializerTag,
    BadDescriptorLength,
    UnknownTextureType,
    UnknownPixelFormat,
    UnknownUsage,
    UnknownResourceOptions,
    UnsupportedProtectionOptions,
    NoMetalDevice,
    ZeroRequirement,
    /// The request names a task id that resolves to no active task, so there is
    /// nowhere to write the reply. Checked by the caller in `runtime/drain/mod.rs`
    /// rather than here — the vocabulary still owns the reason, because the
    /// alternative is one untyped `reason=bad_task` sitting inside an event
    /// family where every other reason is typed.
    BadTask,
}

impl crate::observe::Decline for QueryError {
    /// Every slug carries the `heap_query_` prefix.
    ///
    /// Not decoration: these names are generic enough (`short_payload`,
    /// `unknown_pixel_format`, `unknown_usage`) that they describe checks half
    /// the crate also makes. `unknown_pixel_format` in fact **collided** with
    /// `TranslateReason`'s, and no per-enum uniqueness test could see it — both
    /// enums were internally consistent, and a grep of the fail log for that
    /// slug would have returned a mix of two unrelated subsystems' refusals.
    /// The prefix is what makes the check nameable at crate scope.
    fn slug(&self) -> &'static str {
        match self {
            Self::ShortPayload => "heap_query_short_payload",
            Self::BadReplyLength => "heap_query_bad_reply_length",
            Self::BadSerializerLength => "heap_query_bad_serializer_length",
            Self::UnknownSerializerTag => "heap_query_unknown_serializer_tag",
            Self::BadDescriptorLength => "heap_query_bad_descriptor_length",
            Self::UnknownTextureType => "heap_query_unknown_texture_type",
            Self::UnknownPixelFormat => "heap_query_unknown_pixel_format",
            Self::UnknownUsage => "heap_query_unknown_usage",
            Self::UnknownResourceOptions => "heap_query_unknown_resource_options",
            Self::UnsupportedProtectionOptions => "heap_query_unsupported_protection_options",
            Self::NoMetalDevice => "heap_query_no_metal_device",
            Self::ZeroRequirement => "heap_query_zero_requirement",
            Self::BadTask => "heap_query_bad_task",
        }
    }
}

pub fn decode_request(payload: &[u8]) -> Result<Request, QueryError> {
    if payload.len() < REQUEST_HEADER_LEN {
        return Err(QueryError::ShortPayload);
    }
    let task_id = ld32(&payload[0..]);
    let reply_gva = ld64(&payload[4..]);
    let reply_len = ld64(&payload[12..]);
    if reply_gva == 0 || reply_len < REPLY_LEN as u64 {
        return Err(QueryError::BadReplyLength);
    }
    let serializer_len = ld32(&payload[20..]) as usize;
    if serializer_len != SERIALIZED_TEXTURE_LEN
        || payload.len() != REQUEST_HEADER_LEN + serializer_len
    {
        return Err(QueryError::BadSerializerLength);
    }
    let serialized = &payload[REQUEST_HEADER_LEN..];
    if ld32(serialized) != SERIALIZED_TEXTURE_TAG {
        return Err(QueryError::UnknownSerializerTag);
    }
    if ld32(&serialized[4..]) as usize != SERIALIZED_TEXTURE_LEN
        || serialized.len() != SERIALIZED_TEXTURE_LEN
    {
        return Err(QueryError::BadDescriptorLength);
    }
    let body = &serialized[8..];
    let descriptor = decode_serialized_texture_descriptor(body)?;
    Ok(Request {
        task_id,
        reply_gva,
        reply_len,
        descriptor,
    })
}

/// Decode the shared 32-byte `PGSerializedTextureDescriptor` body.
///
/// The same body is embedded in heap-texture resource opcode `0x15` and in the
/// buffer-backed opcode 9; keeping one decoder prevents the query and resource
/// paths from drifting.
///
/// Read through `reims_vgpu_wire`'s view rather than at offsets restated here,
/// so a field this device names is the field Apple's bytes derived. Two of the
/// names below are this device's reading and not the wire crate's:
/// `framebuffer_only` and `is_drawable` are `packed[5:4]`, which that crate
/// carries as `unidentified_flags` because neither is an `MTLTextureDescriptor`
/// property and no perturbation has moved either. They read 0 on every fixture.
pub fn decode_serialized_texture_descriptor(body: &[u8]) -> Result<TextureDescriptor, QueryError> {
    if body.len() != TEXTURE_BODY_LEN {
        return Err(QueryError::BadDescriptorLength);
    }
    let d: &wire::TextureDescriptorBody =
        reims_vgpu_wire::view(body).map_err(|_| QueryError::BadDescriptorLength)?;
    let packed = d.packed.get();
    Ok(TextureDescriptor {
        texture_type: d.texture_type(),
        framebuffer_only: packed & (1 << 4) != 0,
        is_drawable: packed & (1 << 5) != 0,
        allow_gpu_optimized_contents: d.allow_gpu_optimized_contents(),
        usage: d.usage() as u32,
        pixel_format: d.pixel_format(),
        width: d.width.get(),
        height: d.height.get(),
        depth: d.depth.get(),
        mipmap_level_count: d.mipmap_level_count.get(),
        sample_count: d.sample_count.get(),
        array_length: d.array_length.get(),
        resource_options: d.resource_options.get(),
        // The wire crate calls this word `unidentified_u64` and has moved it in
        // no capture. `protection_options` is this device's ported reading of
        // it; `query_size_and_align` refuses a non-zero one rather than acting
        // on the name, which is the only safe thing to do with a field whose
        // meaning nothing has confirmed.
        protection_options: d.unidentified_u64.get(),
        swizzle: None,
    })
}

/// Decode the 40-byte wide `PGSerializedTextureDescriptor` body.
///
/// The wide form is not a longer narrow one: `usage` leaves the packed word for
/// a `u32` of its own and four swizzle ordinals trail, so every field after the
/// first byte sits at a different offset. Which of the two a record carries is
/// a property of its **opcode**, never of its length — see
/// [`crate::runtime::decode::resource::decode_heap_texture`].
///
/// The fortieth byte is declared and never written, so nothing reads it.
pub fn decode_wide_serialized_texture_descriptor(
    body: &[u8],
) -> Result<TextureDescriptor, QueryError> {
    if body.len() != WIDE_TEXTURE_BODY_LEN {
        return Err(QueryError::BadDescriptorLength);
    }
    let d: &wire::WideTextureDescriptorBody =
        reims_vgpu_wire::view(body).map_err(|_| QueryError::BadDescriptorLength)?;
    Ok(TextureDescriptor {
        texture_type: d.texture_type(),
        // The same two bits the narrow form carries. Bit 7 is a third flag that
        // exists only here — the narrow serializer never writes it — and it is
        // unnamed in both crates, so it is not read.
        framebuffer_only: d.type_and_flags & (1 << 4) != 0,
        is_drawable: d.type_and_flags & (1 << 5) != 0,
        allow_gpu_optimized_contents: d.allow_gpu_optimized_contents(),
        usage: d.usage.get(),
        pixel_format: d.pixel_format.get(),
        width: d.width.get(),
        height: d.height.get(),
        depth: d.depth.get(),
        mipmap_level_count: d.mipmap_level_count.get(),
        sample_count: d.sample_count.get(),
        array_length: d.array_length.get(),
        resource_options: d.resource_options.get(),
        protection_options: d.unidentified_u64.get(),
        swizzle: Some([
            d.swizzle_red,
            d.swizzle_green,
            d.swizzle_blue,
            d.swizzle_alpha,
        ]),
    })
}

#[cfg(target_os = "macos")]
pub fn query_size_and_align(desc: &TextureDescriptor) -> Result<SizeAndAlign, QueryError> {
    use metal::{
        MTLResourceOptions, MTLTextureType, MTLTextureUsage, TextureDescriptor as MtlDescriptor,
    };
    use objc::runtime::{NO, YES};
    use objc::{msg_send, sel, sel_impl};

    let texture_type = match desc.texture_type {
        0 => MTLTextureType::D1,
        1 => MTLTextureType::D1Array,
        2 => MTLTextureType::D2,
        3 => MTLTextureType::D2Array,
        4 => MTLTextureType::D2Multisample,
        5 => MTLTextureType::Cube,
        6 => MTLTextureType::CubeArray,
        7 => MTLTextureType::D3,
        8 => MTLTextureType::D2MultisampleArray,
        _ => return Err(QueryError::UnknownTextureType),
    };
    let pixel_format =
        pixel_format_from_wire(desc.pixel_format).ok_or(QueryError::UnknownPixelFormat)?;
    let usage = MTLTextureUsage::from_bits(desc.usage as u64).ok_or(QueryError::UnknownUsage)?;
    let resource_options = MTLResourceOptions::from_bits(desc.resource_options as u64)
        .ok_or(QueryError::UnknownResourceOptions)?;
    if desc.protection_options != 0 {
        return Err(QueryError::UnsupportedProtectionOptions);
    }
    let device = metal::Device::system_default().ok_or(QueryError::NoMetalDevice)?;
    let mtl = MtlDescriptor::new();
    mtl.set_texture_type(texture_type);
    mtl.set_pixel_format(pixel_format);
    mtl.set_width(desc.width as u64);
    mtl.set_height(desc.height as u64);
    mtl.set_depth(desc.depth as u64);
    mtl.set_mipmap_level_count(desc.mipmap_level_count as u64);
    mtl.set_sample_count(desc.sample_count as u64);
    mtl.set_array_length(desc.array_length as u64);
    mtl.set_resource_options(resource_options);
    mtl.set_usage(usage);
    unsafe {
        let framebuffer_only = if desc.framebuffer_only { YES } else { NO };
        let is_drawable = if desc.is_drawable { YES } else { NO };
        let allow_gpu_optimized = if desc.allow_gpu_optimized_contents {
            YES
        } else {
            NO
        };
        let _: () = msg_send![&*mtl, setFramebufferOnly: framebuffer_only];
        let _: () = msg_send![&*mtl, setIsDrawable: is_drawable];
        let _: () = msg_send![
            &*mtl,
            setAllowGPUOptimizedContents: allow_gpu_optimized
        ];
    }
    let requirement = device.heap_texture_size_and_align(&mtl);
    if requirement.size == 0 || requirement.align == 0 {
        return Err(QueryError::ZeroRequirement);
    }
    Ok(SizeAndAlign {
        size: requirement.size,
        align: requirement.align,
    })
}

#[cfg(not(target_os = "macos"))]
pub fn query_size_and_align(_desc: &TextureDescriptor) -> Result<SizeAndAlign, QueryError> {
    // The Linux Vulkan pathway does not yet have a verified equivalence between
    // VkImage memory requirements and Apple's guest heap placement contract.
    Err(QueryError::NoMetalDevice)
}

#[cfg(target_os = "macos")]
fn pixel_format_from_wire(raw: u16) -> Option<metal::MTLPixelFormat> {
    use metal::MTLPixelFormat as F;
    Some(match raw as u64 {
        x if x == F::Invalid as u64 => F::Invalid,
        x if x == F::A8Unorm as u64 => F::A8Unorm,
        x if x == F::R8Unorm as u64 => F::R8Unorm,
        x if x == F::R8Unorm_sRGB as u64 => F::R8Unorm_sRGB,
        x if x == F::R8Snorm as u64 => F::R8Snorm,
        x if x == F::R8Uint as u64 => F::R8Uint,
        x if x == F::R8Sint as u64 => F::R8Sint,
        x if x == F::R16Unorm as u64 => F::R16Unorm,
        x if x == F::R16Snorm as u64 => F::R16Snorm,
        x if x == F::R16Uint as u64 => F::R16Uint,
        x if x == F::R16Sint as u64 => F::R16Sint,
        x if x == F::R16Float as u64 => F::R16Float,
        x if x == F::RG8Unorm as u64 => F::RG8Unorm,
        x if x == F::RG8Unorm_sRGB as u64 => F::RG8Unorm_sRGB,
        x if x == F::RG8Snorm as u64 => F::RG8Snorm,
        x if x == F::RG8Uint as u64 => F::RG8Uint,
        x if x == F::RG8Sint as u64 => F::RG8Sint,
        x if x == F::B5G6R5Unorm as u64 => F::B5G6R5Unorm,
        x if x == F::A1BGR5Unorm as u64 => F::A1BGR5Unorm,
        x if x == F::ABGR4Unorm as u64 => F::ABGR4Unorm,
        x if x == F::BGR5A1Unorm as u64 => F::BGR5A1Unorm,
        x if x == F::R32Uint as u64 => F::R32Uint,
        x if x == F::R32Sint as u64 => F::R32Sint,
        x if x == F::R32Float as u64 => F::R32Float,
        x if x == F::RG16Unorm as u64 => F::RG16Unorm,
        x if x == F::RG16Snorm as u64 => F::RG16Snorm,
        x if x == F::RG16Uint as u64 => F::RG16Uint,
        x if x == F::RG16Sint as u64 => F::RG16Sint,
        x if x == F::RG16Float as u64 => F::RG16Float,
        x if x == F::RGBA8Unorm as u64 => F::RGBA8Unorm,
        x if x == F::RGBA8Unorm_sRGB as u64 => F::RGBA8Unorm_sRGB,
        x if x == F::RGBA8Snorm as u64 => F::RGBA8Snorm,
        x if x == F::RGBA8Uint as u64 => F::RGBA8Uint,
        x if x == F::RGBA8Sint as u64 => F::RGBA8Sint,
        x if x == F::BGRA8Unorm as u64 => F::BGRA8Unorm,
        x if x == F::BGRA8Unorm_sRGB as u64 => F::BGRA8Unorm_sRGB,
        x if x == F::RGB10A2Unorm as u64 => F::RGB10A2Unorm,
        x if x == F::RGB10A2Uint as u64 => F::RGB10A2Uint,
        x if x == F::RG11B10Float as u64 => F::RG11B10Float,
        x if x == F::RGB9E5Float as u64 => F::RGB9E5Float,
        x if x == F::BGR10A2Unorm as u64 => F::BGR10A2Unorm,
        x if x == F::RG32Uint as u64 => F::RG32Uint,
        x if x == F::RG32Sint as u64 => F::RG32Sint,
        x if x == F::RG32Float as u64 => F::RG32Float,
        x if x == F::RGBA16Unorm as u64 => F::RGBA16Unorm,
        x if x == F::RGBA16Snorm as u64 => F::RGBA16Snorm,
        x if x == F::RGBA16Uint as u64 => F::RGBA16Uint,
        x if x == F::RGBA16Sint as u64 => F::RGBA16Sint,
        x if x == F::RGBA16Float as u64 => F::RGBA16Float,
        x if x == F::RGBA32Uint as u64 => F::RGBA32Uint,
        x if x == F::RGBA32Sint as u64 => F::RGBA32Sint,
        x if x == F::RGBA32Float as u64 => F::RGBA32Float,
        x if x == F::Depth16Unorm as u64 => F::Depth16Unorm,
        x if x == F::Depth32Float as u64 => F::Depth32Float,
        x if x == F::Stencil8 as u64 => F::Stencil8,
        x if x == F::Depth24Unorm_Stencil8 as u64 => F::Depth24Unorm_Stencil8,
        x if x == F::Depth32Float_Stencil8 as u64 => F::Depth32Float_Stencil8,
        x if x == F::X32_Stencil8 as u64 => F::X32_Stencil8,
        x if x == F::X24_Stencil8 as u64 => F::X24_Stencil8,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn live_request_fixture() -> Vec<u8> {
        let words = [
            0x1u32, 0x162200, 0x0, 0x10, 0x0, 0x28, 0x16, 0x28, 0x7d0342, 0xb4, 0x87, 0x1, 0x10001,
            0x200001, 0x0, 0x0,
        ];
        words.into_iter().flat_map(u32::to_le_bytes).collect()
    }

    #[test]
    fn decodes_live_heap_texture_query() {
        let request = decode_request(&live_request_fixture()).unwrap();
        assert_eq!(request.task_id, 1);
        assert_eq!(request.reply_gva, 0x162200);
        assert_eq!(request.reply_len, 16);
        assert_eq!(
            request.descriptor,
            TextureDescriptor {
                texture_type: 2,
                framebuffer_only: false,
                is_drawable: false,
                allow_gpu_optimized_contents: true,
                usage: 3,
                pixel_format: 125,
                width: 180,
                height: 135,
                depth: 1,
                mipmap_level_count: 1,
                sample_count: 1,
                array_length: 1,
                resource_options: 0x20,
                protection_options: 0,
                // The narrow body has no swizzle field at all. `None` rather
                // than the identity: see [`TextureDescriptor::swizzle`].
                swizzle: None,
            }
        );
    }

    #[test]
    fn rejects_unknown_serializer_version() {
        let mut payload = live_request_fixture();
        payload[24..28].copy_from_slice(&0x99u32.to_le_bytes());
        assert_eq!(
            decode_request(&payload),
            Err(QueryError::UnknownSerializerTag)
        );
    }

    #[test]
    fn encodes_two_u64_reply_fields() {
        let reply = SizeAndAlign {
            size: 0x78000,
            align: 0x80,
        }
        .encode();
        assert_eq!(ld64(&reply[0..]), 0x78000);
        assert_eq!(ld64(&reply[8..]), 0x80);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_query_returns_nonzero_requirement() {
        let request = decode_request(&live_request_fixture()).unwrap();
        let result = query_size_and_align(&request.descriptor).unwrap();
        assert!(result.size >= 180 * 135 * 16);
        assert!(result.align.is_power_of_two());
    }
}
