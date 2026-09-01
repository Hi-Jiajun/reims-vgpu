//! `MTLPixelFormat` → `VkFormat`, including the sRGB transfer function and the
//! component mapping Vulkan needs where its channel set differs from Metal's.
//!
//! # Why this table is total
//!
//! Before it, the same Metal→Vulkan pixel decision was re-made at each call
//! site that needed one, and the sRGB qualifier was folded into its linear
//! sibling at twelve independent sites with no record that anything was lost.
//! A lost qualifier looked exactly like a supported format. Here every
//! contract-defined `MTL_FORMAT_*` value has exactly one arm, `*_SRGB` formats
//! reach their `VK_FORMAT_*_SRGB` counterpart, and anything else declines by
//! name through [`Refusal::UnknownPixelFormat`].
//!
//! # sRGB is a choice, not an accident
//!
//! A path that genuinely cannot apply the transfer function (because it is
//! moving raw texels, not shading) asks for [`PixelFormat::linear_vk`], and the
//! doors that hand a linear sibling back say so in their own answer — see
//! [`Sampled::srgb_lost`]. The loss is then one grep away instead of invisible.
//!
//! # What this module is not allowed to know
//!
//! It maps one vocabulary onto another and holds no state, no handle and no
//! host capability. Whether *this* device can sample, render to or store a
//! format it names is a capability question the rail answers elsewhere; whether
//! a role admits the format at all is a contract question
//! `pixel_format::render_target_bpp` / `storage_selector` / `sampled_class`
//! already answer. Folding either into this table would make one guest format
//! mean two things depending on where it was asked.

use ash::vk;

use reims_vgpu_core::pixel_format::{
    self, ColorNumericType, SwizzlePlan, SwizzleSource, TexelLayout, COMPONENT_A, COMPONENT_B,
    COMPONENT_G, COMPONENT_R,
};

/// Why a pixel-format translation did not happen.
///
/// Three reasons, and they are three because they are three different bugs: a
/// value no contract defines, a format this rail carries no guest byte layout
/// for, and a format the contract says is not a colour attachment. A single
/// "unsupported format" would put all three under one slug and make the fail
/// log unable to say which.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Not a value the decode contract defines.
    UnknownPixelFormat(u16),
    /// Defined, but this rail carries no guest texel layout for it. A
    /// *different* answer from an undefined value, so the fail log distinguishes
    /// "we do not know this format" from "this rail does not carry it".
    NoSampledLayout(u16),
    /// Defined, and the contract does not make it a colour attachment.
    NoColorAttachmentFormat(u16),
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::UnknownPixelFormat(_) => "unknown_pixel_format",
            Self::NoSampledLayout(_) => "no_sampled_layout",
            Self::NoColorAttachmentFormat(_) => "no_color_attachment_format",
        }
    }

    /// The Metal format the refusal is about.
    #[must_use]
    pub const fn format(self) -> u16 {
        match self {
            Self::UnknownPixelFormat(f)
            | Self::NoSampledLayout(f)
            | Self::NoColorAttachmentFormat(f) => f,
        }
    }
}

/// A guest texel layout, the channel plan that reaches it, and whether getting
/// there dropped the sRGB transfer function.
///
/// The loss is a field rather than a separate return, because a caller that can
/// ignore it is a caller that has silently changed the colour of a frame. What
/// the loss is *called* on the failure channel belongs to the caller, which is
/// the only side that knows which rail took it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sampled {
    pub layout: TexelLayout,
    /// The guest format was sRGB-encoded and this layout is not. The caller
    /// owes its census a downgrade.
    pub srgb_lost: bool,
    pub components: SwizzlePlan,
}

/// Whether a format's stored values carry the sRGB electro-optical transfer
/// function, which the hardware applies on sample and reverses on write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferFunction {
    Linear,
    Srgb,
}

/// One decoded Metal pixel format, expressed in Vulkan terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelFormat {
    /// The Vulkan format that reproduces the Metal format faithfully — sRGB
    /// included. This is what a render target or sampled image should bind
    /// unless something concrete prevents it.
    pub vk: vk::Format,
    /// Same channel order, same bit layout, linear transfer function. Equals
    /// [`Self::vk`] for a format that was already linear. Binding this for an
    /// sRGB Metal format is a **downgrade**, and the call site that takes it
    /// owes its census one — [`Sampled::srgb_lost`] is the same fact where a
    /// door already answers it.
    pub linear_vk: vk::Format,
    pub transfer: TransferFunction,
    /// Bytes per texel in guest linear storage, from the decode contract — the
    /// single source, so the two can never drift.
    pub bytes_per_texel: u32,
    /// How Metal's presented `(r,g,b,a)` channels sit on this Vulkan format's
    /// channels. Identity for every format whose channel set Vulkan matches;
    /// non-identity only where the Vulkan 1.2 baseline has no equivalent
    /// format (see `A8Unorm`).
    pub components: SwizzlePlan,
}

impl PixelFormat {
    pub fn is_srgb(&self) -> bool {
        matches!(self.transfer, TransferFunction::Srgb)
    }
}

pub const IDENTITY: SwizzlePlan = SwizzlePlan {
    source: [
        SwizzleSource::R,
        SwizzleSource::G,
        SwizzleSource::B,
        SwizzleSource::A,
    ],
};

/// Metal `A8Unorm` presents `(0, 0, 0, a)`. The Vulkan 1.2 baseline has no
/// single-channel alpha format — `VK_FORMAT_A8_UNORM_KHR` arrived with
/// `VK_KHR_maintenance5`, well above the floor every matrix cell must meet — so
/// the byte rides in `R8_UNORM` and this mapping puts it back in alpha with the
/// colour channels zeroed. Identical to what the CPU texel path already
/// produces for this format.
pub const ALPHA_IN_RED: SwizzlePlan = SwizzlePlan {
    source: [
        SwizzleSource::Zero,
        SwizzleSource::Zero,
        SwizzleSource::Zero,
        SwizzleSource::R,
    ],
};

fn linear(vk: vk::Format, bytes_per_texel: u32) -> PixelFormat {
    PixelFormat {
        vk,
        linear_vk: vk,
        transfer: TransferFunction::Linear,
        bytes_per_texel,
        components: IDENTITY,
    }
}

fn srgb(vk: vk::Format, linear_vk: vk::Format, bytes_per_texel: u32) -> PixelFormat {
    PixelFormat {
        vk,
        linear_vk,
        transfer: TransferFunction::Srgb,
        bytes_per_texel,
        components: IDENTITY,
    }
}

/// Translate one decoded `MTLPixelFormat`.
///
/// Total over the values `reims_vgpu_core::pixel_format` defines; every other
/// value declines by name rather than reaching a default. Depth/stencil arms
/// are included because the same enum carries them on the wire — whether a
/// given role (colour attachment, storage image, sampled) admits a format is a
/// *contract* question answered by `render_target_bpp` / `storage_selector` /
/// `sampled_class`, and a *device* question answered by the capability layer;
/// neither is this function's job.
pub fn translate(mtl: u16) -> Result<PixelFormat, Refusal> {
    use pixel_format as p;
    Ok(match mtl {
        p::MTL_FORMAT_A8_UNORM => PixelFormat {
            components: ALPHA_IN_RED,
            ..linear(vk::Format::R8_UNORM, 1)
        },
        p::MTL_FORMAT_R8_UNORM => linear(vk::Format::R8_UNORM, 1),
        p::MTL_FORMAT_R8_UINT => linear(vk::Format::R8_UINT, 1),
        p::MTL_FORMAT_R16_UNORM => linear(vk::Format::R16_UNORM, 2),
        p::MTL_FORMAT_R16_FLOAT => linear(vk::Format::R16_SFLOAT, 2),
        p::MTL_FORMAT_RG8_UNORM => linear(vk::Format::R8G8_UNORM, 2),
        p::MTL_FORMAT_RG8_UINT => linear(vk::Format::R8G8_UINT, 2),
        p::MTL_FORMAT_RG16_UNORM => linear(vk::Format::R16G16_UNORM, 4),
        p::MTL_FORMAT_RG16_UINT => linear(vk::Format::R16G16_UINT, 4),
        p::MTL_FORMAT_R32_UINT => linear(vk::Format::R32_UINT, 4),
        p::MTL_FORMAT_R32_SINT => linear(vk::Format::R32_SINT, 4),
        p::MTL_FORMAT_R32_FLOAT => linear(vk::Format::R32_SFLOAT, 4),
        p::MTL_FORMAT_RG16_FLOAT => linear(vk::Format::R16G16_SFLOAT, 4),
        p::MTL_FORMAT_RGBA8_UNORM => linear(vk::Format::R8G8B8A8_UNORM, 4),
        p::MTL_FORMAT_RGBA8_UNORM_SRGB => {
            srgb(vk::Format::R8G8B8A8_SRGB, vk::Format::R8G8B8A8_UNORM, 4)
        }
        p::MTL_FORMAT_RGBA8_UINT => linear(vk::Format::R8G8B8A8_UINT, 4),
        p::MTL_FORMAT_RGBA8_SINT => linear(vk::Format::R8G8B8A8_SINT, 4),
        p::MTL_FORMAT_BGRA8_UNORM => linear(vk::Format::B8G8R8A8_UNORM, 4),
        p::MTL_FORMAT_BGRA8_UNORM_SRGB => {
            srgb(vk::Format::B8G8R8A8_SRGB, vk::Format::B8G8R8A8_UNORM, 4)
        }
        p::MTL_FORMAT_RGB9E5_FLOAT => linear(vk::Format::E5B9G9R9_UFLOAT_PACK32, 4),
        // The BC block-compressed families. `bytes_per_texel` here is bytes per
        // **4x4 block** — 8 or 16 — which is what `pixel_format::block_geometry`
        // says and what every sizing expression on the sampled rail asks for.
        // The uncompressed arms above are the same field with a 1x1 block, so
        // this is not a second meaning; it is the same number with the grid
        // stated. See `pixel_format::MTL_FORMAT_BC1_RGBA` for why the family
        // arrives whole and `caps::device_features::DeviceFeatures::
        // texture_compression_bc` for the one feature that gates all of it.
        p::MTL_FORMAT_BC1_RGBA => linear(vk::Format::BC1_RGBA_UNORM_BLOCK, p::BC_BLOCK_BYTES_8),
        p::MTL_FORMAT_BC1_RGBA_SRGB => srgb(
            vk::Format::BC1_RGBA_SRGB_BLOCK,
            vk::Format::BC1_RGBA_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_8,
        ),
        p::MTL_FORMAT_BC2_RGBA => linear(vk::Format::BC2_UNORM_BLOCK, p::BC_BLOCK_BYTES_16),
        p::MTL_FORMAT_BC2_RGBA_SRGB => srgb(
            vk::Format::BC2_SRGB_BLOCK,
            vk::Format::BC2_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_16,
        ),
        p::MTL_FORMAT_BC3_RGBA => linear(vk::Format::BC3_UNORM_BLOCK, p::BC_BLOCK_BYTES_16),
        p::MTL_FORMAT_BC3_RGBA_SRGB => srgb(
            vk::Format::BC3_SRGB_BLOCK,
            vk::Format::BC3_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_16,
        ),
        p::MTL_FORMAT_BC4_R_UNORM => linear(vk::Format::BC4_UNORM_BLOCK, p::BC_BLOCK_BYTES_8),
        p::MTL_FORMAT_BC4_R_SNORM => linear(vk::Format::BC4_SNORM_BLOCK, p::BC_BLOCK_BYTES_8),
        p::MTL_FORMAT_BC5_RG_UNORM => linear(vk::Format::BC5_UNORM_BLOCK, p::BC_BLOCK_BYTES_16),
        p::MTL_FORMAT_BC5_RG_SNORM => linear(vk::Format::BC5_SNORM_BLOCK, p::BC_BLOCK_BYTES_16),
        p::MTL_FORMAT_BC6H_RGB_FLOAT => linear(vk::Format::BC6H_SFLOAT_BLOCK, p::BC_BLOCK_BYTES_16),
        p::MTL_FORMAT_BC6H_RGB_UFLOAT => {
            linear(vk::Format::BC6H_UFLOAT_BLOCK, p::BC_BLOCK_BYTES_16)
        }
        p::MTL_FORMAT_BC7_RGBA_UNORM => linear(vk::Format::BC7_UNORM_BLOCK, p::BC_BLOCK_BYTES_16),
        p::MTL_FORMAT_BC7_RGBA_UNORM_SRGB => srgb(
            vk::Format::BC7_SRGB_BLOCK,
            vk::Format::BC7_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_16,
        ),
        // The packed 32-bit colour family. Each Vulkan spelling is the same
        // word cut the same way as its Metal one — `A2B10G10R10` puts red in
        // the low bits as `RGB10A2Unorm` does, `A2R10G10B10` puts blue there as
        // `BGR10A2Unorm` does, and `B10G11R11` is `RG11B10Float`'s word — so a
        // guest texel is sampled unchanged rather than converted.
        p::MTL_FORMAT_RGB10A2_UNORM => linear(vk::Format::A2B10G10R10_UNORM_PACK32, 4),
        p::MTL_FORMAT_BGR10A2_UNORM => linear(vk::Format::A2R10G10B10_UNORM_PACK32, 4),
        p::MTL_FORMAT_RG11B10_FLOAT => linear(vk::Format::B10G11R11_UFLOAT_PACK32, 4),
        // `RGB10A2Uint` has no arm on purpose: an integer texel must not run
        // through the unorm converters, so it is declared for its width in the
        // decode contract and refused by name here, as `R8Uint` and `RG8Uint`
        // are.
        p::MTL_FORMAT_RGBA16_UNORM => linear(vk::Format::R16G16B16A16_UNORM, 8),
        p::MTL_FORMAT_RGBA16_UINT => linear(vk::Format::R16G16B16A16_UINT, 8),
        p::MTL_FORMAT_RGBA16_FLOAT => linear(vk::Format::R16G16B16A16_SFLOAT, 8),
        p::MTL_FORMAT_RGBA32_UINT => linear(vk::Format::R32G32B32A32_UINT, 16),
        p::MTL_FORMAT_RGBA32_FLOAT => linear(vk::Format::R32G32B32A32_SFLOAT, 16),
        p::MTL_FORMAT_DEPTH16_UNORM => linear(vk::Format::D16_UNORM, 2),
        p::MTL_FORMAT_DEPTH32_FLOAT => linear(vk::Format::D32_SFLOAT, 4),
        p::MTL_FORMAT_STENCIL8 => linear(vk::Format::S8_UINT, 1),
        p::MTL_FORMAT_DEPTH24_UNORM_STENCIL8 => linear(vk::Format::D24_UNORM_S8_UINT, 4),
        p::MTL_FORMAT_DEPTH32_FLOAT_STENCIL8 => linear(vk::Format::D32_SFLOAT_S8_UINT, 8),
        // Metal's `X*_Stencil8` are stencil-only *views* of the combined
        // depth-stencil cell, not distinct storage: the decode contract already
        // gives them the same cell size and stencil offset as the format they
        // view (`depth_stencil_packing`). Vulkan has no stencil-only view
        // format either, so they translate to the combined format and the
        // STENCIL aspect selects the plane. This is the contract's own layout,
        // not an invented fallback.
        p::MTL_FORMAT_X32_STENCIL8 => linear(vk::Format::D32_SFLOAT_S8_UINT, 8),
        p::MTL_FORMAT_X24_STENCIL8 => linear(vk::Format::D24_UNORM_S8_UINT, 4),
        other => return Err(Refusal::UnknownPixelFormat(other)),
    })
}

/// Whether a decoded Metal pixel format carries the sRGB transfer function.
///
/// Delegates to the decode contract so the crate has exactly one answer, and a
/// unit test holds the two in agreement.
pub fn is_srgb(mtl: u16) -> bool {
    pixel_format::is_srgb(mtl)
}

/// The guest texel layout for a decoded Metal pixel format, and the decline to
/// record if reaching it dropped the sRGB qualifier.
///
/// `Ok((layout, Some(reason)))` means the layout is right but the transfer
/// function was lost; `Ok((layout, None))` means nothing was lost. A format the
/// contract defines but this rail carries no layout for declines with
/// [`Refusal::NoSampledLayout`] — a *different* slug from an undefined
/// wire value, so the fail log distinguishes "we do not know this format" from
/// "this rail does not carry it".
///
/// The answer is a contract [`TexelLayout`], not a Vulkan format, because its
/// callers are the CPU-upload and in-place-gather rails in `runtime/`: they
/// reason about how many bytes a guest texel occupies and in which channel
/// order, which is a decode question. The host spelling of that layout is
/// [`vk_texel_layout`], applied once where the engine builds the image.
///
/// Callers still choose which layouts they accept: a rail that only handles
/// four-byte texels asks [`TexelLayout::is_four_byte_color`] rather than a
/// narrower entry point, so this table stays the single Metal-side rule.
pub fn sampled_pixels(mtl: u16) -> Result<Sampled, Refusal> {
    let f = translate(mtl)?;
    // The compressed families answer from the contract rather than from a
    // `linear_vk` arm here, because `runtime::draw::texture_view` needs the same
    // answer and cannot reach this module. One mapping, asked twice — see
    // `pixel_format::block_compressed_layout`. Whether this host can sample it
    // is a capability the rail carries, not a fact of the translation.
    if let Some(layout) = pixel_format::block_compressed_layout(mtl) {
        return Ok(Sampled {
            layout,
            srgb_lost: false,
            components: pixel_format::swizzle_identity(),
        });
    }
    // A format whose Metal channels do not sit identically on its Vulkan
    // channels needs a component mapping on the view to sample correctly.
    let layout = match f.linear_vk {
        vk::Format::R8G8B8A8_UNORM => TexelLayout::Rgba8,
        vk::Format::B8G8R8A8_UNORM => TexelLayout::Bgra8,
        vk::Format::R8_UNORM => TexelLayout::R8,
        vk::Format::R8G8_UNORM => TexelLayout::Rg8,
        // Single-channel float rides its own native rail (color-management
        // LUTs). `R16_SFLOAT` is a spec-mandatory sampled+linear format, so it
        // is unconditional. `R32_SFLOAT`'s linear-filter feature is optional
        // (absent on Apple/MoltenVK): the layout is named here (a decode fact),
        // but the rail that emits it must confirm the host can filter it — see
        // `try_linear_sample_zero_copy`'s `supports_sampled_r32f_linear_filter`
        // gate — or the sample stays fail-visible.
        vk::Format::R16_SFLOAT => TexelLayout::R16Float,
        vk::Format::R32_SFLOAT => TexelLayout::R32Float,
        // The ten-bit biplanar video planes, native for the reason the float
        // layouts above are native. Both are Vulkan-mandatory sampled formats
        // with mandatory linear filtering, so neither needs a capability gate.
        vk::Format::R16_UNORM => TexelLayout::R16Unorm,
        vk::Format::R16G16_UNORM => TexelLayout::Rg16Unorm,
        vk::Format::R16G16_UINT => TexelLayout::Rg16Uint,
        vk::Format::R32G32B32A32_SFLOAT => TexelLayout::Rgba32Float,
        // The half-float colour layouts. A recent macOS window server
        // composites in `MTLPixelFormatRGBA16Float`, and every such bind used to
        // land on the CPU re-read rung and be quantized to unorm8 on the way in
        // — 99 % of this rail's format declines on a driven macos-26 boot. Both
        // are exact as guest bytes: the Metal and Vulkan spellings are the same
        // little-endian binary16 channels in the same order.
        vk::Format::R16G16B16A16_UNORM => TexelLayout::Rgba16Unorm,
        vk::Format::R16G16B16A16_SFLOAT => TexelLayout::Rgba16Float,
        vk::Format::R16G16_SFLOAT => TexelLayout::Rg16Float,
        // The packed 32-bit colour layouts, native for the reason the wide
        // layouts above are: the word is the guest's, bit for bit, and no
        // conversion to unorm8 could cut a ten- or eleven-bit channel without
        // discarding what the guest picked the format for.
        vk::Format::A2B10G10R10_UNORM_PACK32 => TexelLayout::Rgb10a2Unorm,
        vk::Format::A2R10G10B10_UNORM_PACK32 => TexelLayout::Bgr10a2Unorm,
        vk::Format::B10G11R11_UFLOAT_PACK32 => TexelLayout::Rg11b10Float,
        _ => return Err(Refusal::NoSampledLayout(mtl)),
    };
    // The format's own channel plan travels with the layout instead of being a
    // reason to refuse. A byte layout says how wide a texel is and in what
    // order its bytes sit; it cannot say that `A8Unorm`'s byte belongs in alpha
    // rather than red, which is why this used to decline every non-identity
    // format outright.
    //
    // Returning it puts the obligation on the caller and the compiler enforces
    // it: a rail that can fold this into its image view's component mapping
    // does so, and one that cannot must decline by name. Deriving it at a call
    // site instead is not available — the plan is a property of the *Metal*
    // format, and a rail holding only a `TexelLayout` or a `VkFormat` has
    // already lost the distinction between `A8Unorm` and `R8Unorm`.
    Ok(Sampled {
        layout,
        srgb_lost: f.is_srgb(),
        components: f.components,
    })
}

/// The Vulkan format for a guest [`TexelLayout`].
///
/// The single crossing from the decode vocabulary to the host one, applied
/// where the engine creates a sampled image. Linear by construction: a layout
/// carries no transfer function, so a rail that reaches a sampled image through
/// here has already recorded whatever [`sampled_pixels`] handed back.
pub fn vk_texel_layout(layout: TexelLayout) -> vk::Format {
    match layout {
        TexelLayout::Rgba8 => vk::Format::R8G8B8A8_UNORM,
        TexelLayout::Bgra8 => vk::Format::B8G8R8A8_UNORM,
        TexelLayout::R8 => vk::Format::R8_UNORM,
        TexelLayout::Rg8 => vk::Format::R8G8_UNORM,
        TexelLayout::R16Float => vk::Format::R16_SFLOAT,
        TexelLayout::R32Float => vk::Format::R32_SFLOAT,
        TexelLayout::R16Unorm => vk::Format::R16_UNORM,
        TexelLayout::Rg16Unorm => vk::Format::R16G16_UNORM,
        TexelLayout::Rg16Uint => vk::Format::R16G16_UINT,
        TexelLayout::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        TexelLayout::Rgba16Unorm => vk::Format::R16G16B16A16_UNORM,
        TexelLayout::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        TexelLayout::Rg16Float => vk::Format::R16G16_SFLOAT,
        TexelLayout::Rgb10a2Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
        TexelLayout::Bgr10a2Unorm => vk::Format::A2R10G10B10_UNORM_PACK32,
        TexelLayout::Rg11b10Float => vk::Format::B10G11R11_UFLOAT_PACK32,
        // The BC families. Each Metal spelling and its Vulkan counterpart are
        // the same block layout with the same bytes in the same order, so the
        // guest's payload is uploaded verbatim — which is why these need no
        // conversion arm anywhere and are admitted as one family.
        TexelLayout::Bc1Rgba => vk::Format::BC1_RGBA_UNORM_BLOCK,
        TexelLayout::Bc2Rgba => vk::Format::BC2_UNORM_BLOCK,
        TexelLayout::Bc3Rgba => vk::Format::BC3_UNORM_BLOCK,
        TexelLayout::Bc4RUnorm => vk::Format::BC4_UNORM_BLOCK,
        TexelLayout::Bc4RSnorm => vk::Format::BC4_SNORM_BLOCK,
        TexelLayout::Bc5RgUnorm => vk::Format::BC5_UNORM_BLOCK,
        TexelLayout::Bc5RgSnorm => vk::Format::BC5_SNORM_BLOCK,
        TexelLayout::Bc6hRgbFloat => vk::Format::BC6H_SFLOAT_BLOCK,
        TexelLayout::Bc6hRgbUfloat => vk::Format::BC6H_UFLOAT_BLOCK,
        TexelLayout::Bc7Rgba => vk::Format::BC7_UNORM_BLOCK,
    }
}

/// The sRGB spelling of a guest [`TexelLayout`], for the layouts that have one.
///
/// The counterpart of [`vk_texel_layout`] for an image whose stored values are
/// sRGB-encoded, so the hardware decodes on sample. `None` for every layout that
/// cannot hold an sRGB image — see [`TexelLayout::has_srgb_encoding`], which
/// this agrees with by a `const` assertion below rather than by a second list.
///
/// Written as the inverse of [`storage_format`] and held to it by
/// `the_srgb_spelling_of_a_layout_stores_that_layout`: a pair that disagreed
/// would key a resident allocation on one format and bind a view of the other.
pub fn srgb_texel_layout(layout: TexelLayout) -> Option<vk::Format> {
    match layout {
        TexelLayout::Rgba8 => Some(vk::Format::R8G8B8A8_SRGB),
        TexelLayout::Bgra8 => Some(vk::Format::B8G8R8A8_SRGB),
        // The four BC families Apple gives an sRGB spelling. BC4/BC5 are
        // single- and two-channel data rather than colour and BC6H is HDR
        // float, so none of the three has one on either side of the boundary.
        TexelLayout::Bc1Rgba => Some(vk::Format::BC1_RGBA_SRGB_BLOCK),
        TexelLayout::Bc2Rgba => Some(vk::Format::BC2_SRGB_BLOCK),
        TexelLayout::Bc3Rgba => Some(vk::Format::BC3_SRGB_BLOCK),
        TexelLayout::Bc7Rgba => Some(vk::Format::BC7_SRGB_BLOCK),
        _ => None,
    }
}

/// The [`TexelLayout`] a Vulkan format is, or `None` for a format that is not
/// one of them.
///
/// The inverse of [`vk_texel_layout`], for the engine, which holds a resolved
/// `vk::Format` for an attachment and needs the layout to ask
/// [`reims_vgpu_core::pixel_format`] how to write a texel of it. Written as a
/// search of `TexelLayout::ALL` rather than as a second `match`, so it cannot
/// disagree with the forward map and a new layout is covered the moment it is
/// added to `ALL`.
pub fn texel_layout_of(format: vk::Format) -> Option<TexelLayout> {
    // A layout describes stored bytes, and the transfer function does not change
    // them, so the fold is [`storage_format`]'s and is not spelled twice.
    let format = storage_format(format);
    TexelLayout::ALL
        .iter()
        .copied()
        .find(|&l| vk_texel_layout(l) == format)
}

/// The format an image is *allocated* in, for a requested view format.
///
/// A Metal texture view over an `IOSurface` is a second interpretation of one
/// allocation, never a second allocation. `BGRA8Unorm` and `BGRA8Unorm_sRGB`
/// name the same stored bytes and differ only in the fixed-function conversion
/// applied on render writes and sampled reads. Vulkan expresses exactly that
/// with one `VkImage` created `MUTABLE_FORMAT` and one `VkImageView` per
/// interpretation, so the allocation is keyed on this format and the transfer
/// function rides on the view.
///
/// **Folding here is what keeps one surface to one resident.** Keying an
/// allocation on the view format instead forks the resident the moment a guest
/// binds one surface through both spellings — which the guest does — and the two
/// images then alternate frame to frame, each holding half the content. That is
/// a content defect, not a colour one, and it is why this fold is not optional.
pub fn storage_format(format: vk::Format) -> vk::Format {
    match format {
        vk::Format::R8G8B8A8_SRGB => vk::Format::R8G8B8A8_UNORM,
        vk::Format::B8G8R8A8_SRGB => vk::Format::B8G8R8A8_UNORM,
        // The four BC families with an sRGB spelling. Same rule one storage
        // shape over: a compressed image's blocks are identical bytes under
        // either qualifier, so both spellings must resolve to one allocation
        // and differ only in the view. `the_srgb_spelling_of_a_layout_stores_
        // that_layout` is what holds this to `srgb_texel_layout`.
        vk::Format::BC1_RGBA_SRGB_BLOCK => vk::Format::BC1_RGBA_UNORM_BLOCK,
        vk::Format::BC2_SRGB_BLOCK => vk::Format::BC2_UNORM_BLOCK,
        vk::Format::BC3_SRGB_BLOCK => vk::Format::BC3_UNORM_BLOCK,
        vk::Format::BC7_SRGB_BLOCK => vk::Format::BC7_UNORM_BLOCK,
        other => other,
    }
}

/// The two formats one resident image answers for, derived from the single
/// format the guest declared so they cannot disagree.
///
/// A resident is asked its format by two kinds of caller and they want two
/// different answers:
///
/// * **the allocation** — what `vkCreateImage` is given, what keys reuse of a
///   live slot, and what buckets the image in the recycle pool. Two declarations
///   that differ only in transfer function are one `MTLTexture` seen through two
///   `newTextureViewWithPixelFormat:` views, so they must resolve to one image.
///   That is [`storage_format`]'s rule, and this is where it is applied.
/// * **the declaration** — what a render pass attaches, and the stronger of the
///   two answers a sampled bind can be given, because it carries the transfer
///   function Vulkan applies on write and on read.
///
/// Both were spelled `color_format`, one `vk::Format` doing both jobs, and the
/// two `registry_ensure*` arms picked differently: the primary one keyed the
/// allocation on the declaration, which forks one surface into two images the
/// moment the guest binds it through both spellings; the secondary one keyed
/// reuse on the allocation while registering, creating and recycling under the
/// declaration, so an sRGB resident there was retired on every ensure and its
/// recycled image went into a bucket nothing takes from. Carrying the pair as
/// one value is what makes those two mistakes unspellable: neither answer can be
/// reached without naming which one it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentFormat(vk::Format);

impl ResidentFormat {
    /// The resident behind a guest declaration of `declared`.
    pub fn of(declared: vk::Format) -> Self {
        Self(declared)
    }

    /// What the guest declared: the render pass's attachment format, and the
    /// interpretation a sampled view of this resident decodes through.
    pub fn declared(self) -> vk::Format {
        self.0
    }

    /// The allocation family. Keys `vkCreateImage`, live-slot reuse and the
    /// recycle bucket — never the view.
    pub fn allocation(self) -> vk::Format {
        storage_format(self.0)
    }

    /// Whether the declaration adds a transfer function the allocation does not
    /// carry, so the attachment needs a view of its own over the same image.
    pub fn needs_own_view(self) -> bool {
        self.declared() != self.allocation()
    }
}

/// The format a **sampled** view of a resident must be created with, given the
/// format the bind asked for and the format the resident itself holds.
///
/// # Why the bind's own answer is not enough
///
/// A sampled bind's format is resolved from a [`TexelLayout`] — see
/// `engine::SampledResource::format` — and a layout names stored bytes and so
/// carries no transfer function at all. That is the right vocabulary for the
/// rails that upload guest bytes, because there the layout *is* everything the
/// device knows. It is not enough for the one source that has a second,
/// stronger declaration available: a `SampledSource::Target`, whose resident was
/// created by [`color_attachment`] from the guest's own `MTLPixelFormat` and
/// therefore does carry the transfer function.
///
/// Without this fold a resident the guest declared `BGRA8Unorm_sRGB` is written
/// through a `B8G8R8A8_SRGB` attachment — Vulkan encodes linear to sRGB, as
/// Metal does — and then sampled through a `B8G8R8A8_UNORM` view, which decodes
/// nothing. The still-encoded value is composited and encoded a second time by
/// the next attachment write, so the frame carries **exactly one sRGB encode too
/// many**. That is a colour defect with no counter behind it: every rail
/// succeeds, nothing declines, and the picture is washed out in the direction
/// `1.055 x^(1/2.4) - 0.055` describes.
///
/// # Why it can only ever add the transfer function
///
/// Two gates, and both are load-bearing.
///
/// [`stored_bytes_agree`] means it fires only where the two spellings differ in
/// nothing but the transfer function. A bind whose channel order or texel width
/// differs from the resident's is left exactly as it asked — that disagreement
/// is a real one and this is not the place to resolve it.
///
/// **The bind must also have had nothing to say.** Only a `requested` that is
/// already its own [`storage_format`] — a spelling with no transfer function on
/// it — is one the resident may answer for. A bind naming `B8G8R8A8_SRGB` over a
/// resident written through its linear view has stated an interpretation, and
/// Metal's contract is that a texture view's pixel format *is* the
/// interpretation for that bind; answering with the allocation's own spelling
/// drops the decode the bind asked for. Without this gate the function does not
/// add a transfer function, it replaces one side's with the other's, and it goes
/// wrong in whichever direction the resident happens to hold — which is how
/// `resident_sample_uses_the_bindings_compatible_format_view` caught it.
///
/// What is left unresolved, and said rather than hidden: a bind spelled through
/// a `TexelLayout` is linear because that vocabulary has no other spelling, so
/// this cannot tell it from a guest that genuinely asked for a linear view of an
/// sRGB surface. The resident wins there, which is right for every rail that
/// reaches here through a layout and would be wrong for a rail that could say
/// linear and meant it. Closing that needs the sampled rails to carry the
/// guest's `MTLPixelFormat` rather than a byte layout; it is not closable here.
pub fn sample_view_format(requested: vk::Format, resident: vk::Format) -> vk::Format {
    if requested == storage_format(requested) && stored_bytes_agree(requested, resident) {
        resident
    } else {
        requested
    }
}

/// Whether two Vulkan formats describe the same stored bytes for the same
/// texel, so a transfer that converts nothing may move one into the other.
///
/// This is the question a `vkCmdCopyImageToBuffer` out of a render target into
/// guest pages actually asks, and it is not format equality. The two sides
/// reaching that copy answer two different questions by design: an attachment
/// carries the guest's transfer function, because [`color_attachment`] keeps it
/// so Vulkan performs the fixed-function linear-to-sRGB encode on write, while a
/// guest destination is spelled as a [`TexelLayout`] via [`vk_texel_layout`] and
/// has no transfer function to carry. A guest render target declared
/// `BGRA8Unorm_sRGB` therefore meets itself as `B8G8R8A8_SRGB` against
/// `B8G8R8A8_UNORM`, forever, and equality reads that as a disagreement.
///
/// Vulkan is explicit that it is not one: buffer/image copies perform no format
/// conversion, so what crosses is the stored texel, and [`storage_format`] is
/// this module's existing fold onto it. Everything a byte-level comparison must
/// still separate survives that fold — channel order (`R8G8B8A8` against
/// `B8G8R8A8`) and texel width (eight-bit against half-float) both differ in the
/// storage format, not only in the view.
pub fn stored_bytes_agree(held: vk::Format, want: vk::Format) -> bool {
    storage_format(held) == storage_format(want)
}

/// Whether a Vulkan colour format stores its first and third channels in BGRA
/// order. The transfer function is deliberately irrelevant: UNORM and sRGB
/// views interpret the same four stored bytes.
pub fn has_bgra_order(format: vk::Format) -> bool {
    matches!(texel_layout_of(format), Some(TexelLayout::Bgra8))
}

/// Every Vulkan format a colour attachment may take, and the decline for a
/// format the rail does not render to.
///
/// The result is the resolved [`vk::Format`] rather than an engine enum, so an
/// sRGB target reaches an sRGB attachment and gets Vulkan's fixed-function
/// linear-to-sRGB conversion on writes.
///
/// The narrowing is deliberate and stays. Metal renders to far more formats
/// than this device carries; admitting one the rest of the pass machinery has
/// never carried would trade a named decline for a wrong picture.
///
/// **Which formats those are is the contract's answer, not this function's.**
/// [`pixel_format::render_target_bpp`] says in its own doc that "the match arms
/// *are* the renderable set", and this used to hold a second list — of Vulkan
/// formats rather than Metal ones, so nothing could compare them. They had
/// already drifted: the contract admitted `RGBA16_FLOAT`, which is what lets a
/// half-float *primary* attachment be created at the format the guest declared,
/// while this refused it. One guest format was therefore renderable as slot 0
/// and declined as a secondary MRT slot, which is not a narrowing anybody chose.
///
/// Asking the contract makes the two arms one answer, and makes adding a
/// renderable format a single edit there rather than a pair of edits that a
/// commit can half-land.
pub fn color_attachment(mtl: u16) -> Result<(vk::Format, ColorNumericType), Refusal> {
    let f = translate(mtl)?;
    let numeric = pixel_format::render_target_numeric_type(mtl)
        .ok_or(Refusal::NoColorAttachmentFormat(mtl))?;
    Ok((f.vk, numeric))
}

/// The guest's scanout byte order, in Vulkan terms.
///
/// The compositor's framebuffers are `MTLPixelFormatBGRA8Unorm`, so a resident
/// target and a swapchain image both use this format to keep
/// the present path free of a channel swap. Named once because it is one
/// decision — spelled at each site it would drift, and a single wrong spelling
/// shows up as red-and-blue-swapped output rather than a failure.
///
/// A test holds it equal to what the pixel table answers for
/// `MTL_FORMAT_BGRA8_UNORM`, so it cannot become a second opinion.
pub const SCANOUT_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

/// The engine's neutral resident colour format, used where content is not
/// destined straight for scanout and the channel order does not matter.
pub const RESIDENT_RGBA_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// The transient depth attachment format. Depth-only passes use this; a pass
/// that also needs stencil negotiates a combined format against the device,
/// because which combined format exists is a capability question.
pub const TRANSIENT_DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// Resident colour format for a target, by whether its bytes must already be in
/// guest scanout order.
pub fn resident_color(bgra: bool) -> vk::Format {
    if bgra {
        SCANOUT_FORMAT
    } else {
        RESIDENT_RGBA_FORMAT
    }
}

/// Bytes occupied by one texel of a Vulkan format, for the formats this table
/// can produce.
///
/// The inverse view of [`PixelFormat::bytes_per_texel`], needed once the engine
/// stores a resolved `VkFormat` rather than a byte-layout enum that carried its
/// own size. `None` for anything outside the set — including block-compressed
/// and multi-planar formats, whose footprint is not one number per texel — so a
/// caller declines by name instead of computing a wrong buffer size.
///
/// An sRGB format has the footprint of its linear sibling, which is what makes
/// flipping a rail to sRGB a pure colour-space change with no allocation
/// consequences.
/// Every [`TexelLayout`] answers from the contract's own width, so a layout
/// added to [`TexelLayout::ALL`] is covered here the moment it exists. This was
/// a hand-kept second copy of those widths, and it was missing
/// `R16G16B16A16_UNORM` for as long as that layout had existed — which cost
/// macOS 26 a hundred and eight draws a boot, because a width this table did not
/// know is indistinguishable from a block-compressed one and declines by the
/// same name. Same argument as [`texel_layout_of`] being a search rather than a
/// second `match`.
/// The storage **block** grid of a Vulkan format, for the formats this table can
/// produce.
///
/// [`bytes_per_texel`] with the grid stated, and derived from the same
/// [`texel_layout_of`] search so the two cannot disagree. A caller sizing a
/// linear buffer for an image must ask this rather than `bytes_per_texel`:
/// multiplying a block byte count by width and height over-counts a compressed
/// image by sixteen, which is a refusal against the guest's own correctly-sized
/// buffer rather than a wrong image.
///
/// sRGB spellings fold through [`storage_format`] onto the allocation they share
/// with their linear sibling. That fold is what covers the four `BC*_SRGB_BLOCK`
/// formats, which a sampled bind of an sRGB compressed texture is created as.
pub fn vk_block_geometry(format: vk::Format) -> Option<reims_vgpu_core::extent::BlockGeometry> {
    if let Some(layout) = texel_layout_of(storage_format(format)) {
        return Some(layout.block());
    }
    Some(reims_vgpu_core::extent::BlockGeometry {
        width: 1,
        height: 1,
        bytes: bytes_per_texel(format)?,
    })
}

pub fn bytes_per_texel(format: vk::Format) -> Option<u32> {
    if let Some(layout) = texel_layout_of(format) {
        return Some(layout.bytes_per_texel());
    }
    // What remains is the formats that are deliberately not `TexelLayout`s:
    // depth/stencil, the packed shared-exponent float, and the integer and sRGB
    // spellings of the colour orders. None is a guest linear texel layout, so
    // none has a contract width to derive from.
    Some(match format {
        vk::Format::S8_UINT => 1,
        vk::Format::D16_UNORM => 2,
        vk::Format::R32_UINT
        | vk::Format::R32_SINT
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::R8G8B8A8_UINT
        | vk::Format::R8G8B8A8_SINT
        | vk::Format::B8G8R8A8_SRGB
        | vk::Format::E5B9G9R9_UFLOAT_PACK32
        | vk::Format::D32_SFLOAT
        | vk::Format::D24_UNORM_S8_UINT => 4,
        vk::Format::R16G16B16A16_UINT | vk::Format::D32_SFLOAT_S8_UINT => 8,
        vk::Format::R32G32B32A32_UINT | vk::Format::R32G32B32A32_SFLOAT => 16,
        _ => return None,
    })
}

/// A decoded swizzle plan as the `VkImageView` component mapping that performs
/// it in hardware.
///
/// The plan passed in is **already folded**: the caller composes the decoded
/// texture-view swizzle over the format's own channel remap with
/// [`reims_vgpu_core::pixel_format::SwizzlePlan::after`], because a
/// `VkComponentMapping` can express one plan and a bind may need both. This
/// function does no composing of its own and must not start — it would then be
/// a second place the fold happens, and the two would disagree the first time
/// only one of them was updated.
///
/// It used to be the view swizzle alone, which was safe only because
/// [`sampled_pixels`] refused every format with a non-identity plan. It no
/// longer refuses them: it returns the plan, and `A8Unorm` — whose byte rides
/// in `R8_UNORM` — is sampled rather than sent to the CPU rung.
///
/// This is what makes a swizzled view cost nothing: Vulkan applies the mapping
/// at sample time, so the texels never have to be rewritten on the CPU (which
/// would force the whole texture onto the CPU upload path and cost the
/// zero-copy property for it).
pub fn vk_component_mapping(plan: &SwizzlePlan) -> vk::ComponentMapping {
    fn one(source: SwizzleSource) -> vk::ComponentSwizzle {
        match source {
            SwizzleSource::Zero => vk::ComponentSwizzle::ZERO,
            SwizzleSource::One => vk::ComponentSwizzle::ONE,
            SwizzleSource::R => vk::ComponentSwizzle::R,
            SwizzleSource::G => vk::ComponentSwizzle::G,
            SwizzleSource::B => vk::ComponentSwizzle::B,
            SwizzleSource::A => vk::ComponentSwizzle::A,
        }
    }
    vk::ComponentMapping {
        r: one(plan.source[COMPONENT_R]),
        g: one(plan.source[COMPONENT_G]),
        b: one(plan.source[COMPONENT_B]),
        a: one(plan.source[COMPONENT_A]),
    }
}

/// Whether a Metal format's channels sit identically on its Vulkan format.
///
/// The component plan is a property of the **Metal** format, not of the Vulkan
/// one it resolves to: `A8Unorm` and `R8Unorm` both land on `R8_UNORM`, and
/// only the first needs its byte moved back into alpha. A rail that has already
/// reduced a format to a host format or a byte layout can therefore no longer
/// derive the plan, which is exactly why [`sampled_pixels`] declines a
/// non-identity format instead of admitting one it could not describe.
///
/// The sampled rail relies on that: it takes the plan from [`sampled_pixels`]
/// and folds it under the decoded texture-view swizzle (see [`vk_component_mapping`]),
/// which it can only do because the plan travels with the layout instead of
/// being re-derived downstream. This predicate is now a *reader* of the same
/// fact rather than a gate on it, and a test holds the two in agreement.
pub fn has_identity_components(mtl: u16) -> bool {
    translate(mtl)
        .map(|f| f.components == IDENTITY)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::pixel_format as p;

    /// [`vk_texel_layout`] names stored bytes and so never carries a transfer
    /// function.
    ///
    /// The property matters where a format built by that function is asked a
    /// channel-order question: `gva_resident_format`'s output reaches
    /// `GvaTargetKey` on two sides of the GVA store witness, and both used to
    /// spell the question as `== SCANOUT_FORMAT`. That spelling is only right
    /// while this holds, and it is wrong the moment a caller passes a format
    /// from anywhere else — which is exactly what happened one rail over, in
    /// `engine::ResidentReadSnapshot::bgra`, where the format came from the
    /// attachment and did carry one. Both sites ask [`has_bgra_order`] now; this
    /// says the switch changed no answer.
    #[test]
    fn a_stored_texel_layout_never_names_a_transfer_function() {
        for &layout in TexelLayout::ALL {
            let f = vk_texel_layout(layout);
            assert_eq!(storage_format(f), f, "{layout:?}");
            assert_eq!(has_bgra_order(f), f == SCANOUT_FORMAT, "{layout:?}");
        }
    }

    /// A sampled bind can only ever spell a stored-byte format, so every
    /// resident the guest declared with an sRGB format would be sampled without
    /// its decode if the bind's own answer were taken. `sample_view_format`
    /// restores it, for both channel orders and both spellings of the bind.
    #[test]
    fn a_sampled_view_takes_the_transfer_function_from_the_resident() {
        for (linear, srgb) in [
            (SCANOUT_FORMAT, vk::Format::B8G8R8A8_SRGB),
            (RESIDENT_RGBA_FORMAT, vk::Format::R8G8B8A8_SRGB),
        ] {
            // The defect: the bind names stored bytes, the resident carries the
            // qualifier, and the view has to be the resident's.
            assert_eq!(sample_view_format(linear, srgb), srgb);
            // Already agreed, either way round: nothing to restore.
            assert_eq!(sample_view_format(srgb, srgb), srgb);
            assert_eq!(sample_view_format(linear, linear), linear);
            // A bind that spelled the qualifier itself keeps it. The comment
            // beside this line already said so — "this must not become a
            // downgrade of a bind that already spelled it" — while the assertion
            // under it demanded the downgrade, and the engine obeyed the
            // assertion: a resident written through its linear attachment view
            // and sampled through its sRGB sibling was bound linear and decoded
            // nothing.
            assert_eq!(sample_view_format(srgb, linear), srgb);
        }
    }

    /// The fold may add a transfer function and nothing else. A bind whose
    /// channel order or texel width disagrees with the resident is a real
    /// disagreement and is left exactly as it asked — resolving it here would
    /// silently rewrite what the shader samples.
    #[test]
    fn a_sampled_view_never_changes_the_stored_bytes_the_bind_asked_for() {
        for &layout in TexelLayout::ALL {
            let requested = vk_texel_layout(layout);
            for &resident in TexelLayout::ALL {
                for resident in [
                    vk_texel_layout(resident),
                    // Both sRGB spellings a resident can hold, so the case that
                    // matters is exercised against every requested layout.
                    vk::Format::B8G8R8A8_SRGB,
                    vk::Format::R8G8B8A8_SRGB,
                ] {
                    let got = sample_view_format(requested, resident);
                    assert!(
                        stored_bytes_agree(got, requested),
                        "{requested:?} against {resident:?} became {got:?}"
                    );
                }
            }
        }
    }

    /// The copy out of a render target into guest pages converts nothing, so the
    /// only thing the two sides must agree on is the stored texel.
    ///
    /// Every guest render target declared with an sRGB format meets itself
    /// across this comparison — the attachment keeps the transfer function so
    /// Vulkan encodes on write, and the guest destination is spelled as a bare
    /// [`TexelLayout`] — so format equality here refuses a copy whose bytes are
    /// identical. On a driven macos-13 Maps leg that was 1 001 refusals out of
    /// 1 001, and the app's canvas kept the zeros its pages were allocated with.
    #[test]
    fn a_transfer_function_is_not_a_disagreement_about_stored_bytes() {
        for (view, stored) in [
            (vk::Format::B8G8R8A8_SRGB, vk::Format::B8G8R8A8_UNORM),
            (vk::Format::R8G8B8A8_SRGB, vk::Format::R8G8B8A8_UNORM),
        ] {
            assert!(
                stored_bytes_agree(view, stored),
                "{view:?} and {stored:?} are one allocation seen two ways"
            );
            assert!(stored_bytes_agree(stored, view), "the rule is symmetric");
            assert!(stored_bytes_agree(view, view));
        }
    }

    /// What the comparison is *for* survives the fold. Channel order and texel
    /// width are storage facts, not view facts, so folding the transfer function
    /// cannot admit either — a BGRA resident under an RGBA destination would need
    /// an exchange this copy cannot perform, and a half-float destination over an
    /// eight-bit resident would overlap its rows at half their true pitch.
    #[test]
    fn channel_order_and_texel_width_still_disagree() {
        for (a, b) in [
            (vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM),
            (vk::Format::B8G8R8A8_SRGB, vk::Format::R8G8B8A8_UNORM),
            (vk::Format::B8G8R8A8_SRGB, vk::Format::R8G8B8A8_SRGB),
            (vk::Format::B8G8R8A8_UNORM, vk::Format::R16G16B16A16_SFLOAT),
            (vk::Format::B8G8R8A8_SRGB, vk::Format::R16G16B16A16_SFLOAT),
        ] {
            assert!(
                !stored_bytes_agree(a, b),
                "{a:?} and {b:?} do not store the same bytes for one texel"
            );
        }
    }

    /// `texel_layout_of` is `vk_texel_layout` read backwards, for every layout
    /// and for nothing else.
    ///
    /// The round trip is the whole property: the engine holds a resolved
    /// `vk::Format` for an attachment and uses this to ask the contract how to
    /// write a texel of it, so a layout that does not come back is one whose
    /// seed would be staged at the wrong width. Driven from `TexelLayout::ALL`
    /// so a new layout is covered without anyone adding a line.
    #[test]
    fn every_texel_layout_survives_the_round_trip_through_its_vulkan_format() {
        for &layout in TexelLayout::ALL {
            assert_eq!(
                texel_layout_of(vk_texel_layout(layout)),
                Some(layout),
                "{layout:?} does not come back from its own format"
            );
        }
        // A format that is not a texel layout answers `None` rather than the
        // nearest one; depth is the case the engine could plausibly present.
        assert_eq!(texel_layout_of(TRANSIENT_DEPTH_FORMAT), None);
        assert_eq!(texel_layout_of(vk::Format::UNDEFINED), None);
    }

    /// The table is total over the contract: every defined value maps, with the
    /// expected Vulkan format and texel size.
    #[test]
    fn every_contract_pixel_format_translates() {
        for (mtl, vkf, bpt, transfer) in EXPECTED {
            let got = translate(*mtl).unwrap_or_else(|e| panic!("MTL {mtl:#x}: {e:?}"));
            assert_eq!(got.vk, *vkf, "MTL {mtl:#x} vk format");
            assert_eq!(got.bytes_per_texel, *bpt, "MTL {mtl:#x} texel size");
            assert_eq!(got.transfer, *transfer, "MTL {mtl:#x} transfer function");
        }
    }

    /// The texel size this module reports is the decode contract's, not a
    /// second opinion — the drift `byte_size`-beside-`vk_format` was written to
    /// prevent.
    ///
    /// Compared against the contract's **block** size rather than its
    /// bytes-per-texel. For every uncompressed format those are the same number
    /// — the block is 1x1 — and for the BC families only the block form exists,
    /// because a BC1 texel is half a byte and `bytes_per_pixel` says `None` on
    /// purpose. So this is the stronger reading of the same invariant, not a
    /// weakened one.
    #[test]
    fn texel_size_agrees_with_the_decode_contract() {
        for (mtl, _, _, _) in EXPECTED {
            assert_eq!(
                Some(translate(*mtl).unwrap().bytes_per_texel),
                pixel_format::block_geometry(*mtl).map(|block| block.bytes),
                "MTL {mtl:#x}"
            );
        }
    }

    /// The whole point of L1: an sRGB Metal format reaches an sRGB VkFormat.
    /// If this ever fails, the hardware has silently stopped applying the
    /// transfer function on write and blending is happening in the wrong
    /// colour space.
    #[test]
    fn srgb_formats_reach_an_srgb_vk_format() {
        for mtl in [
            p::MTL_FORMAT_RGBA8_UNORM_SRGB,
            p::MTL_FORMAT_BGRA8_UNORM_SRGB,
        ] {
            let f = translate(mtl).unwrap();
            assert!(f.is_srgb(), "MTL {mtl:#x} lost its sRGB classification");
            assert!(
                matches!(f.vk, vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB),
                "MTL {mtl:#x} mapped to non-sRGB {:?}",
                f.vk
            );
        }
    }

    /// An sRGB format's linear sibling keeps the channel order and bit layout —
    /// downgrading may cost the transfer function and nothing else, or the
    /// stored bytes stop meaning the same thing.
    #[test]
    fn the_linear_sibling_keeps_the_channel_order() {
        let rgba = translate(p::MTL_FORMAT_RGBA8_UNORM_SRGB).unwrap();
        assert_eq!(rgba.linear_vk, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(
            rgba.linear_vk,
            translate(p::MTL_FORMAT_RGBA8_UNORM).unwrap().vk
        );
        let bgra = translate(p::MTL_FORMAT_BGRA8_UNORM_SRGB).unwrap();
        assert_eq!(bgra.linear_vk, vk::Format::B8G8R8A8_UNORM);
        assert_eq!(
            bgra.linear_vk,
            translate(p::MTL_FORMAT_BGRA8_UNORM).unwrap().vk
        );
        assert_eq!(rgba.bytes_per_texel, 4);
        assert_eq!(bgra.bytes_per_texel, 4);
    }

    /// An undefined wire value declines by name instead of reaching a default.
    #[test]
    fn an_unknown_format_declines_by_name() {
        let err = translate(0xffff).unwrap_err();
        assert_eq!(err, Refusal::UnknownPixelFormat(0xffff));
        assert_eq!(err.slug(), "unknown_pixel_format");
        assert!(!is_srgb(0xffff));
        assert!(sampled_pixels(0xffff).is_err());
    }

    /// The constant-fold shortcuts stay honest against the full translation —
    /// this module and the decode contract must not hold two opinions about
    /// which formats are sRGB.
    #[test]
    fn is_srgb_tracks_the_translated_transfer_function() {
        for (mtl, _, _, transfer) in EXPECTED {
            assert_eq!(
                is_srgb(*mtl),
                *transfer == TransferFunction::Srgb,
                "MTL {mtl:#x}"
            );
            assert_eq!(
                is_srgb(*mtl),
                pixel_format::is_srgb(*mtl),
                "MTL {mtl:#x} disagrees with the decode contract"
            );
        }
    }

    /// The sRGB spelling of a layout is the same allocation as its linear one,
    /// and exists for exactly the layouts the contract says can hold an sRGB
    /// image.
    ///
    /// Both halves matter and neither is a restatement. The first keeps
    /// [`srgb_texel_layout`] the inverse of [`storage_format`]: a pair that
    /// disagreed would key a resident allocation on one format while binding a
    /// view of the other, which is the fork [`storage_format`]'s own doc
    /// describes. The second keeps it in step with
    /// [`TexelLayout::has_srgb_encoding`], which is what
    /// [`vk_sampled_bytes`] consults to decide whether an sRGB source is
    /// honourable or has to be reported — a layout answering `true` there with
    /// no spelling here would report a downgrade it could have avoided, and one
    /// answering `false` with a spelling here would hide a real one.
    #[test]
    fn the_srgb_spelling_of_a_layout_stores_that_layout() {
        for layout in TexelLayout::ALL.iter().copied() {
            match srgb_texel_layout(layout) {
                Some(srgb) => {
                    assert_eq!(
                        storage_format(srgb),
                        vk_texel_layout(layout),
                        "{layout:?}: the sRGB view must store the linear allocation"
                    );
                    assert!(
                        layout.has_srgb_encoding(),
                        "{layout:?}: spelled sRGB but the contract says it cannot hold one"
                    );
                }
                None => assert!(
                    !layout.has_srgb_encoding(),
                    "{layout:?}: can hold an sRGB image and has no spelling for it"
                ),
            }
        }
    }

    /// The storage fold changes the transfer function and **never** the channel
    /// order.
    ///
    /// This is the property that makes [`storage_format`] safe to key an
    /// allocation on. A fold that swapped `B8G8R8A8_SRGB` onto an `R8G8B8A8`
    /// storage format would put every texel's red and blue in each other's
    /// bytes, which reaches the screen as a hue rotation and nothing in the
    /// engine would refuse it — an image and a view in the same compatibility
    /// class are both valid Vulkan.
    #[test]
    fn the_storage_fold_never_changes_channel_order() {
        for &format in &[
            vk::Format::R8G8B8A8_SRGB,
            vk::Format::B8G8R8A8_SRGB,
            vk::Format::R8G8B8A8_UNORM,
            vk::Format::B8G8R8A8_UNORM,
        ] {
            let storage = storage_format(format);
            assert_eq!(
                has_bgra_order(storage),
                has_bgra_order(format),
                "{format:?} changed channel order on the way to {storage:?}"
            );
            assert_eq!(
                texel_layout_of(storage),
                texel_layout_of(format),
                "{format:?} changed byte layout on the way to {storage:?}"
            );
        }
        assert_eq!(
            storage_format(vk::Format::B8G8R8A8_SRGB),
            vk::Format::B8G8R8A8_UNORM
        );
        assert_eq!(
            storage_format(vk::Format::R8G8B8A8_SRGB),
            vk::Format::R8G8B8A8_UNORM
        );
    }

    /// Every format the forward map produces is already a storage format, and
    /// the fold is idempotent.
    ///
    /// Together these say an allocation keyed on [`storage_format`] has exactly
    /// one spelling per compatibility class, which is what stops one surface
    /// from being resident twice.
    #[test]
    fn the_storage_fold_is_idempotent_and_closed_over_the_forward_map() {
        for &layout in TexelLayout::ALL {
            let format = vk_texel_layout(layout);
            assert_eq!(
                storage_format(format),
                format,
                "{layout:?} maps to {format:?}, which is not its own storage format"
            );
        }
        for &format in &[
            vk::Format::R8G8B8A8_SRGB,
            vk::Format::B8G8R8A8_SRGB,
            vk::Format::R16G16B16A16_SFLOAT,
        ] {
            let once = storage_format(format);
            assert_eq!(storage_format(once), once, "{format:?} folds twice");
        }
    }

    /// A resident's two answers are the two questions the registry asks, and
    /// only the sRGB pair may separate them.
    ///
    /// The second half is what makes the type cheap: on every format this
    /// device renders to except the two sRGB spellings, the allocation and the
    /// declaration are the same `vk::Format`, so no extra view is ever created
    /// and `needs_own_view` answers false. A change that made some third format
    /// fold would show up here as a new pair rather than as a silent extra view
    /// per resident.
    #[test]
    fn a_residents_allocation_and_declaration_part_only_on_the_transfer_function() {
        for (declared, allocation) in [
            (vk::Format::B8G8R8A8_SRGB, vk::Format::B8G8R8A8_UNORM),
            (vk::Format::R8G8B8A8_SRGB, vk::Format::R8G8B8A8_UNORM),
        ] {
            let f = ResidentFormat::of(declared);
            assert_eq!(f.declared(), declared);
            assert_eq!(f.allocation(), allocation);
            assert!(f.needs_own_view(), "{declared:?}");
            // The two spellings of one surface reach one allocation, which is
            // the whole reason the pair exists.
            assert_eq!(ResidentFormat::of(allocation).allocation(), allocation);
        }
        for layout in TexelLayout::ALL {
            let f = ResidentFormat::of(vk_texel_layout(*layout));
            assert_eq!(f.allocation(), f.declared(), "{layout:?}");
            assert!(!f.needs_own_view(), "{layout:?}");
        }
    }

    /// Every guest texel layout has a Vulkan-side width, and it is the contract's.
    ///
    /// This is the check that was missing. `bytes_per_texel` used to be a second,
    /// hand-kept copy of `TexelLayout::bytes_per_texel`, and when `Rgba16Unorm`
    /// was added to the contract nothing made this side learn it. A `None` here
    /// is not a quiet wrong answer — it is the same verdict a block-compressed
    /// format gets, so the draw is refused by name
    /// (`vk_draw_validate_sampled_no_linear_texel_footprint`) and the guest
    /// silently loses it. macOS 26 lost 108 draws a boot to exactly that.
    ///
    /// Asserting equality with the contract rather than a literal is the point:
    /// a literal table here is what created the drift in the first place.
    #[test]
    fn every_texel_layout_has_the_contract_width_on_the_vulkan_side() {
        for &layout in TexelLayout::ALL {
            let vk = vk_texel_layout(layout);
            assert_eq!(
                bytes_per_texel(vk),
                Some(layout.bytes_per_texel()),
                "{layout:?} ({vk:?}) disagrees with the contract width"
            );
        }
    }

    /// A format that is genuinely not one texel-width answers `None`, so the
    /// derivation above did not turn the decline into a wrong number.
    #[test]
    fn a_block_compressed_format_still_has_no_texel_footprint() {
        assert_eq!(bytes_per_texel(vk::Format::BC1_RGB_UNORM_BLOCK), None);
        assert_eq!(bytes_per_texel(vk::Format::G8_B8R8_2PLANE_420_UNORM), None);
    }

    /// A sampled bind's view mapping is the decoded texture-view swizzle and nothing
    /// else. Identity in, identity out — otherwise every ordinary bind would
    /// pay for a feature almost none of them use.
    #[test]
    fn a_sampled_bind_maps_identity_to_identity() {
        let m = vk_component_mapping(&pixel_format::swizzle_identity());
        assert_eq!(m.r, vk::ComponentSwizzle::R);
        assert_eq!(m.g, vk::ComponentSwizzle::G);
        assert_eq!(m.b, vk::ComponentSwizzle::B);
        assert_eq!(m.a, vk::ComponentSwizzle::A);
    }

    /// A non-identity view reaches the hardware unchanged — the swizzle is a
    /// property of the view, not of the byte order underneath it.
    #[test]
    fn a_sampled_bind_carries_the_view_swizzle() {
        // Read `(b, g, r, 1)`.
        let view = pixel_format::swizzle_plan(&[4, 3, 2, 1]).unwrap();
        let m = vk_component_mapping(&view);
        assert_eq!(m.r, vk::ComponentSwizzle::B);
        assert_eq!(m.g, vk::ComponentSwizzle::G);
        assert_eq!(m.b, vk::ComponentSwizzle::R);
        assert_eq!(m.a, vk::ComponentSwizzle::ONE);
    }

    /// Every `MTLPixelFormat` the decode contract defines, with the Vulkan
    /// format and texel size it must produce. Written out literally rather than
    /// derived from the table under test, so a mistranslation shows up as a
    /// diff instead of agreeing with itself.
    const EXPECTED: &[(u16, vk::Format, u32, TransferFunction)] = &[
        (
            p::MTL_FORMAT_A8_UNORM,
            vk::Format::R8_UNORM,
            1,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_R8_UNORM,
            vk::Format::R8_UNORM,
            1,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_R8_UINT,
            vk::Format::R8_UINT,
            1,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_R16_FLOAT,
            vk::Format::R16_SFLOAT,
            2,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RG8_UNORM,
            vk::Format::R8G8_UNORM,
            2,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RG8_UINT,
            vk::Format::R8G8_UINT,
            2,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_R32_UINT,
            vk::Format::R32_UINT,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_R32_SINT,
            vk::Format::R32_SINT,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_R32_FLOAT,
            vk::Format::R32_SFLOAT,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RG16_FLOAT,
            vk::Format::R16G16_SFLOAT,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RGBA8_UNORM,
            vk::Format::R8G8B8A8_UNORM,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RGBA8_UNORM_SRGB,
            vk::Format::R8G8B8A8_SRGB,
            4,
            TransferFunction::Srgb,
        ),
        (
            p::MTL_FORMAT_RGBA8_UINT,
            vk::Format::R8G8B8A8_UINT,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RGBA8_SINT,
            vk::Format::R8G8B8A8_SINT,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BGRA8_UNORM,
            vk::Format::B8G8R8A8_UNORM,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BGRA8_UNORM_SRGB,
            vk::Format::B8G8R8A8_SRGB,
            4,
            TransferFunction::Srgb,
        ),
        (
            p::MTL_FORMAT_RGB10A2_UNORM,
            vk::Format::A2B10G10R10_UNORM_PACK32,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RG11B10_FLOAT,
            vk::Format::B10G11R11_UFLOAT_PACK32,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BGR10A2_UNORM,
            vk::Format::A2R10G10B10_UNORM_PACK32,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RGB9E5_FLOAT,
            vk::Format::E5B9G9R9_UFLOAT_PACK32,
            4,
            TransferFunction::Linear,
        ),
        // The BC families. The width column is bytes per 4x4 **block** for
        // these, which is what `pixel_format::block_geometry` says and what the
        // sampled rail sizes rows and images from; the uncompressed rows above
        // are the same field with a 1x1 block.
        (
            p::MTL_FORMAT_BC1_RGBA,
            vk::Format::BC1_RGBA_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_8,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC1_RGBA_SRGB,
            vk::Format::BC1_RGBA_SRGB_BLOCK,
            p::BC_BLOCK_BYTES_8,
            TransferFunction::Srgb,
        ),
        (
            p::MTL_FORMAT_BC2_RGBA,
            vk::Format::BC2_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC2_RGBA_SRGB,
            vk::Format::BC2_SRGB_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Srgb,
        ),
        (
            p::MTL_FORMAT_BC3_RGBA,
            vk::Format::BC3_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC3_RGBA_SRGB,
            vk::Format::BC3_SRGB_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Srgb,
        ),
        (
            p::MTL_FORMAT_BC4_R_UNORM,
            vk::Format::BC4_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_8,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC4_R_SNORM,
            vk::Format::BC4_SNORM_BLOCK,
            p::BC_BLOCK_BYTES_8,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC5_RG_UNORM,
            vk::Format::BC5_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC5_RG_SNORM,
            vk::Format::BC5_SNORM_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC6H_RGB_FLOAT,
            vk::Format::BC6H_SFLOAT_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC6H_RGB_UFLOAT,
            vk::Format::BC6H_UFLOAT_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC7_RGBA_UNORM,
            vk::Format::BC7_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC7_RGBA_UNORM_SRGB,
            vk::Format::BC7_SRGB_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Srgb,
        ),
        (
            p::MTL_FORMAT_R16_UNORM,
            vk::Format::R16_UNORM,
            2,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RG16_UNORM,
            vk::Format::R16G16_UNORM,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RG16_UINT,
            vk::Format::R16G16_UINT,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RGBA16_UNORM,
            vk::Format::R16G16B16A16_UNORM,
            8,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RGBA16_UINT,
            vk::Format::R16G16B16A16_UINT,
            8,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RGBA16_FLOAT,
            vk::Format::R16G16B16A16_SFLOAT,
            8,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RGBA32_UINT,
            vk::Format::R32G32B32A32_UINT,
            16,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RGBA32_FLOAT,
            vk::Format::R32G32B32A32_SFLOAT,
            16,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_DEPTH16_UNORM,
            vk::Format::D16_UNORM,
            2,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_DEPTH32_FLOAT,
            vk::Format::D32_SFLOAT,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_STENCIL8,
            vk::Format::S8_UINT,
            1,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_DEPTH24_UNORM_STENCIL8,
            vk::Format::D24_UNORM_S8_UINT,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_DEPTH32_FLOAT_STENCIL8,
            vk::Format::D32_SFLOAT_S8_UINT,
            8,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_X32_STENCIL8,
            vk::Format::D32_SFLOAT_S8_UINT,
            8,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_X24_STENCIL8,
            vk::Format::D24_UNORM_S8_UINT,
            4,
            TransferFunction::Linear,
        ),
    ];

    /// [`EXPECTED`] names every format [`translate`] accepts.
    ///
    /// Every other test here iterates `EXPECTED`, so a format added to
    /// `translate` and not to `EXPECTED` is simply never swept — its texel
    /// width, its transfer function, its channel plan and its membership of
    /// each rail's accepted set all go unchecked, and every test still passes.
    /// `MTLPixelFormatRGBA16Unorm` was added to `translate` in the same commit
    /// as this test, and without it nothing would have noticed either way.
    ///
    /// Swept over the whole `u16` domain rather than over a list of constants,
    /// because a list is the thing being checked. This is a derivation, not a
    /// second spelling: `translate` is the authority on what it accepts and it
    /// is asked about every value it could be given.
    #[test]
    fn expected_names_every_format_the_table_translates() {
        let listed: std::collections::BTreeSet<u16> =
            EXPECTED.iter().map(|(mtl, ..)| *mtl).collect();
        let translated: std::collections::BTreeSet<u16> = (0..=u16::MAX)
            .filter(|mtl| translate(*mtl).is_ok())
            .collect();
        assert_eq!(
            translated, listed,
            "translate accepts formats EXPECTED does not name (or the reverse), so they are \
             swept by no test in this module"
        );
    }
}
