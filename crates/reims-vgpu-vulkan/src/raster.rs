//! The fixed-function render state a guest sets, translated — and the two
//! pieces of it that are host capabilities rather than mappings.
//!
//! # The Y axis points the other way
//!
//! Metal's clip space has +Y up. Vulkan's framebuffer coordinates have +Y
//! down, so the same vertices rasterize vertically mirrored unless something
//! flips them. There are three places that flip could go — the shader, the
//! projection the guest supplied, or the viewport — and only one of them is
//! this device's to touch.
//!
//! So it is the viewport: a negative height with the origin moved to the
//! bottom edge. `VK_KHR_maintenance1` made that legal and Vulkan 1.1 promoted
//! it, so on a 1.2 baseline it is always available and needs no gate. Flipping
//! in the shader would mean rewriting translated guest code; flipping the
//! projection would mean recognising one, and a guest that renders without a
//! conventional projection matrix would be silently wrong.
//!
//! [`viewport`] is therefore the one place the flip happens, and its test is
//! that the bottom of the guest's rectangle is where the negative height
//! points.
//!
//! # Two states are capabilities and not mappings
//!
//! `MTLDepthClipModeClamp` needs `depthClamp`, and `MTLTriangleFillModeLines`
//! needs `fillModeNonSolid`. Both are optional Vulkan features, and neither
//! has a substitute: clipping where the guest asked to clamp throws away
//! geometry it expected to keep, and filling where it asked for lines draws
//! solid triangles over a wireframe. Both refuse by name, and both refuse only
//! for the mode that needs the feature — the default mode of each is always
//! available, so a host without either feature still runs every guest that
//! does not ask.
//!
//! # Four of these states are encoder commands, and Vulkan bakes them
//!
//! Metal sets the cull mode, the winding, the fill mode and the depth clip
//! mode on the *encoder*: `setCullMode:` and its three siblings may be called
//! between two draws of one pass, and the second draw rasterizes differently
//! with the same pipeline state object. Vulkan puts all four in
//! `VkPipelineRasterizationStateCreateInfo`, where changing one means a
//! different pipeline.
//!
//! Taken literally that is a pipeline per state change — a guest that toggles
//! culling around a draw compiles a second pipeline for it, and compilation is
//! the most expensive thing this rail does. The way out is the dynamic state
//! Vulkan offers for exactly these members, so [`RasterCell`] carries the
//! three feature bits that reach them and [`plan`] spends them:
//!
//! - `extendedDynamicState`, core in 1.3, reaches `vkCmdSetCullMode` and
//!   `vkCmdSetFrontFace`. One bit, two states, which is why the cell has one
//!   field for the pair.
//! - `extendedDynamicState3PolygonMode` reaches `vkCmdSetPolygonModeEXT`.
//! - `extendedDynamicState3DepthClampEnable` reaches
//!   `vkCmdSetDepthClampEnableEXT`.
//!
//! A state this host made dynamic is **normalized out of**
//! [`RasterizationState`] and carried in [`DynamicRaster`] instead. That is the
//! whole benefit: if the guest's value stayed in the pipeline state it would
//! still be a cache dimension, two draws differing only in a dynamic cull mode
//! would still hash apart, and nothing would have been saved. The same
//! normalization is why [`blend`](crate::blend) fixes a disabled attachment's
//! factors and [`depth_stencil`](crate::depth_stencil) fixes a disabled
//! stencil's faces.
//!
//! **Making a state dynamic does not make its capability free.** A device
//! without `fillModeNonSolid` refuses `vkCmdSetPolygonModeEXT(LINE)` exactly
//! as it refuses the pipeline member, and the same holds for `depthClamp`. So
//! the two refusals below run on the guest's ordinal whether or not the state
//! is dynamic, and the dynamic path is a placement decision made after the
//! translation, never instead of it.
//!
//! # Depth bias is always enabled
//!
//! `setDepthBias:slopeScale:clamp:` is an encoder command with no enable bit:
//! Metal has no way to say "this pipeline cannot be biased". Vulkan does, and
//! it is a pipeline member — `depthBiasEnable` clear makes
//! `vkCmdSetDepthBias` a no-op. So the pipeline always sets it, and the values
//! arrive dynamically. Zero bias with the enable on rasterizes identically to
//! the enable off, so this costs a guest that never biases nothing; leaving it
//! clear would silently drop the bias of a guest that does.
//!
//! # An unknown ordinal is not the default
//!
//! Every parse here is a closed set returning `None` outside it. Folding an
//! unrecognised cull mode onto `None` draws back faces the guest culled;
//! folding an unrecognised winding onto clockwise culls the wrong ones. The
//! refusal carries the ordinal.

use ash::vk;
use reims_vgpu_core::render::{ScissorRect, Viewport};

/// `MTLCullMode`.
pub const MTL_CULL_MODE_NONE: u64 = 0;
pub const MTL_CULL_MODE_FRONT: u64 = 1;
pub const MTL_CULL_MODE_BACK: u64 = 2;

/// `MTLWinding`.
pub const MTL_WINDING_CLOCKWISE: u64 = 0;
pub const MTL_WINDING_COUNTER_CLOCKWISE: u64 = 1;

/// `MTLDepthClipMode`.
pub const MTL_DEPTH_CLIP_MODE_CLIP: u64 = 0;
pub const MTL_DEPTH_CLIP_MODE_CLAMP: u64 = 1;

/// `MTLTriangleFillMode`.
pub const MTL_TRIANGLE_FILL_MODE_FILL: u64 = 0;
pub const MTL_TRIANGLE_FILL_MODE_LINES: u64 = 1;

/// The optional Vulkan features two of these states need.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RasterCell {
    /// `VkPhysicalDeviceFeatures::depthClamp`.
    pub depth_clamp: bool,
    /// `VkPhysicalDeviceFeatures::fillModeNonSolid`.
    pub fill_mode_non_solid: bool,
    /// `VkPhysicalDeviceExtendedDynamicStateFeaturesEXT::extendedDynamicState`,
    /// or 1.3 core.
    ///
    /// One field for two states because it is one feature bit: it reaches
    /// `vkCmdSetCullMode` and `vkCmdSetFrontFace` together, and two fields
    /// that a census could only ever set alike would be two spellings of one
    /// fact.
    pub dynamic_cull_and_winding: bool,
    /// `…ExtendedDynamicState3FeaturesEXT::extendedDynamicState3PolygonMode`.
    pub dynamic_polygon_mode: bool,
    /// `…ExtendedDynamicState3FeaturesEXT::extendedDynamicState3DepthClampEnable`.
    pub dynamic_depth_clamp: bool,
}

/// Why a state the guest set cannot be honoured here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// An ordinal outside the closed set. Never folded onto a neighbour: see
    /// the module doc.
    UnknownOrdinal { state: &'static str, ordinal: u64 },
    /// The guest asked to clamp depth and this device cannot.
    NoDepthClamp,
    /// The guest asked for a wireframe fill and this device cannot.
    NoNonSolidFill,
    /// A viewport or scissor dimension outside the fields Vulkan carries.
    OutOfRange { field: &'static str, value: u64 },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::UnknownOrdinal { .. } => "vk_raster_unknown_ordinal",
            Self::NoDepthClamp => "vk_raster_no_depth_clamp",
            Self::NoNonSolidFill => "vk_raster_no_non_solid_fill",
            Self::OutOfRange { .. } => "vk_raster_out_of_range",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownOrdinal { state, ordinal } => {
                write!(f, "{} state={state} ordinal={ordinal}", self.slug())
            }
            Self::NoDepthClamp | Self::NoNonSolidFill => f.write_str(self.slug()),
            Self::OutOfRange { field, value } => {
                write!(f, "{} field={field} value={value}", self.slug())
            }
        }
    }
}

/// Which faces a draw discards.
///
/// # Errors
///
/// [`Refusal::UnknownOrdinal`] outside the closed set.
pub const fn cull_mode(ordinal: u64) -> Result<vk::CullModeFlags, Refusal> {
    Ok(match ordinal {
        MTL_CULL_MODE_NONE => vk::CullModeFlags::NONE,
        MTL_CULL_MODE_FRONT => vk::CullModeFlags::FRONT,
        MTL_CULL_MODE_BACK => vk::CullModeFlags::BACK,
        _ => {
            return Err(Refusal::UnknownOrdinal {
                state: "cull_mode",
                ordinal,
            })
        }
    })
}

/// Which winding is the front face.
///
/// The mapping is direct and stays direct *because* the Y flip is in the
/// viewport. A flip done anywhere else would reverse the winding of every
/// triangle, and the correction would have to be here — two states each
/// half-describing one transform, with nothing to say which is which.
///
/// # Errors
///
/// [`Refusal::UnknownOrdinal`] outside the closed set.
pub const fn front_face(ordinal: u64) -> Result<vk::FrontFace, Refusal> {
    Ok(match ordinal {
        MTL_WINDING_CLOCKWISE => vk::FrontFace::CLOCKWISE,
        MTL_WINDING_COUNTER_CLOCKWISE => vk::FrontFace::COUNTER_CLOCKWISE,
        _ => {
            return Err(Refusal::UnknownOrdinal {
                state: "winding",
                ordinal,
            })
        }
    })
}

/// Whether fragments outside the depth range are clamped or clipped.
///
/// Returns the `depthClampEnable` this state means.
///
/// # Errors
///
/// [`Refusal::UnknownOrdinal`], or [`Refusal::NoDepthClamp`] when the guest
/// asked to clamp on a device without the feature. Clipping instead throws
/// away geometry the guest expected to keep.
pub const fn depth_clamp(ordinal: u64, cell: RasterCell) -> Result<bool, Refusal> {
    match ordinal {
        MTL_DEPTH_CLIP_MODE_CLIP => Ok(false),
        MTL_DEPTH_CLIP_MODE_CLAMP => {
            if cell.depth_clamp {
                Ok(true)
            } else {
                Err(Refusal::NoDepthClamp)
            }
        }
        _ => Err(Refusal::UnknownOrdinal {
            state: "depth_clip_mode",
            ordinal,
        }),
    }
}

/// How a triangle is filled.
///
/// # Errors
///
/// [`Refusal::UnknownOrdinal`], or [`Refusal::NoNonSolidFill`] when the guest
/// asked for lines on a device without the feature. Filling instead draws
/// solid triangles over a wireframe.
pub const fn polygon_mode(ordinal: u64, cell: RasterCell) -> Result<vk::PolygonMode, Refusal> {
    match ordinal {
        MTL_TRIANGLE_FILL_MODE_FILL => Ok(vk::PolygonMode::FILL),
        MTL_TRIANGLE_FILL_MODE_LINES => {
            if cell.fill_mode_non_solid {
                Ok(vk::PolygonMode::LINE)
            } else {
                Err(Refusal::NoNonSolidFill)
            }
        }
        _ => Err(Refusal::UnknownOrdinal {
            state: "triangle_fill_mode",
            ordinal,
        }),
    }
}

/// The four encoder states a guest sets, as ordinals.
///
/// An aggregate rather than four arguments, because four `u64` in a row is a
/// call site that compiles with two of them transposed. Each field is the raw
/// guest ordinal: parsing is [`plan`]'s, and an unrecognised one refuses there
/// rather than being folded here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GuestRasterState {
    pub cull_mode: u64,
    pub winding: u64,
    pub depth_clip_mode: u64,
    pub fill_mode: u64,
}

impl GuestRasterState {
    /// What an encoder rasterizes with before the guest sets anything.
    ///
    /// Metal's own defaults, and every one of them is the mode that needs no
    /// optional feature — which is what makes a guest that sets none of these
    /// runnable on the barest device this rail admits.
    pub const DEFAULT: Self = Self {
        cull_mode: MTL_CULL_MODE_NONE,
        winding: MTL_WINDING_CLOCKWISE,
        depth_clip_mode: MTL_DEPTH_CLIP_MODE_CLIP,
        fill_mode: MTL_TRIANGLE_FILL_MODE_FILL,
    };
}

impl Default for GuestRasterState {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Which rasterization members this host supplies per draw rather than per
/// pipeline.
///
/// Derived from [`RasterCell`] and never assembled by hand, so the flags and
/// the values in [`DynamicRaster`] cannot disagree about which is which.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RasterDynamic {
    /// `vkCmdSetCullMode` and `vkCmdSetFrontFace`. One flag, because one
    /// feature bit reaches both.
    pub cull_and_winding: bool,
    pub polygon_mode: bool,
    pub depth_clamp_enable: bool,
}

impl RasterDynamic {
    /// What this host offers, from the cell.
    #[must_use]
    pub const fn of(cell: RasterCell) -> Self {
        Self {
            cull_and_winding: cell.dynamic_cull_and_winding,
            polygon_mode: cell.dynamic_polygon_mode,
            depth_clamp_enable: cell.dynamic_depth_clamp,
        }
    }

    /// Nothing dynamic: every state is baked.
    pub const NONE: Self = Self {
        cull_and_winding: false,
        polygon_mode: false,
        depth_clamp_enable: false,
    };

    /// The states a pipeline built with these flags must declare.
    ///
    /// `DEPTH_BIAS` is unconditional, and is here rather than in the caller's
    /// list for the reason the module doc gives: the pipeline always enables
    /// biasing, so it must always take the values dynamically. The viewport
    /// and scissor are the caller's — they are set per pass, not per
    /// rasterizer state, and this type would be claiming a decision it does
    /// not make.
    #[must_use]
    pub fn states(self) -> Vec<vk::DynamicState> {
        let mut out = vec![vk::DynamicState::DEPTH_BIAS];
        if self.cull_and_winding {
            out.push(vk::DynamicState::CULL_MODE);
            out.push(vk::DynamicState::FRONT_FACE);
        }
        if self.polygon_mode {
            out.push(vk::DynamicState::POLYGON_MODE_EXT);
        }
        if self.depth_clamp_enable {
            out.push(vk::DynamicState::DEPTH_CLAMP_ENABLE_EXT);
        }
        out
    }
}

/// The rasterization state a pipeline is created with.
///
/// Spelled out rather than held as the ash structure, which is not `Eq` — and
/// a pipeline key that cannot be compared is one whose cache cannot be
/// asserted.
///
/// **A member this host made dynamic carries its default here, not the
/// guest's value.** See the module doc: the guest's value is in
/// [`DynamicRaster`], and leaving it here as well would keep it a cache
/// dimension and save nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RasterizationState {
    pub depth_clamp_enable: bool,
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    /// Always true. See the module doc.
    pub depth_bias_enable: bool,
    /// Which members above are placeholders because the encoder supplies them.
    pub dynamic: RasterDynamic,
}

impl RasterizationState {
    pub const fn native(self) -> vk::PipelineRasterizationStateCreateInfo<'static> {
        vk::PipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: vk::PipelineRasterizationStateCreateFlags::empty(),
            depth_clamp_enable: as_bool32(self.depth_clamp_enable),
            // Never. Metal has no equivalent, and discarding before
            // rasterization would drop every fragment of every draw.
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: self.polygon_mode,
            cull_mode: self.cull_mode,
            front_face: self.front_face,
            depth_bias_enable: as_bool32(self.depth_bias_enable),
            // Zero, and dynamic: the encoder's `setDepthBias:` supplies all
            // three, and zero is what a guest that never calls it means.
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            // Metal has no line-width state at all, so one is not a default
            // standing in for something — it is the only width this API can
            // ask for, and the only width a device without `wideLines` may be
            // given.
            line_width: 1.0,
            _marker: core::marker::PhantomData,
        }
    }
}

const fn as_bool32(value: bool) -> vk::Bool32 {
    if value {
        vk::TRUE
    } else {
        vk::FALSE
    }
}

/// The guest's values for the members this host made dynamic.
///
/// `Some` exactly where [`RasterizationState`] holds a placeholder, so a
/// caller that sets every `Some` has reproduced the guest's state and a caller
/// that sets none of them is a bug this type can name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DynamicRaster {
    pub cull_mode: Option<vk::CullModeFlags>,
    pub front_face: Option<vk::FrontFace>,
    pub polygon_mode: Option<vk::PolygonMode>,
    pub depth_clamp_enable: Option<bool>,
}

/// A translated rasterizer state: what the pipeline is built with, and what
/// the encoder sets before each draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub state: RasterizationState,
    pub dynamic: DynamicRaster,
}

/// The rasterization members a pipeline bakes when nothing is dynamic.
///
/// Also what a dynamic member is normalized to, which is why it is one
/// constant and not two: a placeholder that drifted from the baked default
/// would make an all-static host and an all-dynamic host build different
/// pipelines for the same guest state.
const BAKED_DEFAULT: (bool, vk::PolygonMode, vk::CullModeFlags, vk::FrontFace) = (
    false,
    vk::PolygonMode::FILL,
    vk::CullModeFlags::NONE,
    vk::FrontFace::CLOCKWISE,
);

/// Translate a guest rasterizer state against this host.
///
/// Every ordinal is parsed and every capability is checked, whether or not
/// the state it names ends up dynamic — see the module doc for why making a
/// state dynamic does not make its feature free.
///
/// # Errors
///
/// [`Refusal::UnknownOrdinal`] for an ordinal outside a closed set, and
/// [`Refusal::NoDepthClamp`] or [`Refusal::NoNonSolidFill`] for a mode this
/// device has no feature for. Nothing is partially translated.
pub fn plan(guest: GuestRasterState, cell: RasterCell) -> Result<Plan, Refusal> {
    // All four first, so a refusal is decided before any placement is. A host
    // that made a state dynamic must still refuse the ordinal it cannot serve.
    let cull = cull_mode(guest.cull_mode)?;
    let winding = front_face(guest.winding)?;
    let clamp = depth_clamp(guest.depth_clip_mode, cell)?;
    let fill = polygon_mode(guest.fill_mode, cell)?;

    let dynamic = RasterDynamic::of(cell);
    let (default_clamp, default_fill, default_cull, default_winding) = BAKED_DEFAULT;
    Ok(Plan {
        state: RasterizationState {
            depth_clamp_enable: if dynamic.depth_clamp_enable {
                default_clamp
            } else {
                clamp
            },
            polygon_mode: if dynamic.polygon_mode {
                default_fill
            } else {
                fill
            },
            cull_mode: if dynamic.cull_and_winding {
                default_cull
            } else {
                cull
            },
            front_face: if dynamic.cull_and_winding {
                default_winding
            } else {
                winding
            },
            depth_bias_enable: true,
            dynamic,
        },
        dynamic: DynamicRaster {
            cull_mode: dynamic.cull_and_winding.then_some(cull),
            front_face: dynamic.cull_and_winding.then_some(winding),
            polygon_mode: dynamic.polygon_mode.then_some(fill),
            depth_clamp_enable: dynamic.depth_clamp_enable.then_some(clamp),
        },
    })
}

/// The viewport a guest rectangle becomes, with the Y flip in it.
///
/// The origin moves to the bottom edge and the height is negated, which is
/// what makes +Y-up clip space rasterize the way the guest drew it. See the
/// module doc for why the flip is here and not in the shader or the
/// projection.
///
/// # Errors
///
/// [`Refusal::OutOfRange`] for a dimension outside the `f32` fields Vulkan
/// carries. The guest's are doubles.
pub fn viewport(guest: Viewport) -> Result<vk::Viewport, Refusal> {
    let x = f64::from_bits(guest.origin_x_bits);
    let y = f64::from_bits(guest.origin_y_bits);
    let width = f64::from_bits(guest.width_bits);
    let height = f64::from_bits(guest.height_bits);
    let near = f64::from_bits(guest.z_near_bits);
    let far = f64::from_bits(guest.z_far_bits);
    for (field, value) in [
        ("origin_x", x),
        ("origin_y", y),
        ("width", width),
        ("height", height),
    ] {
        if !value.is_finite() {
            return Err(Refusal::OutOfRange {
                field,
                value: value.to_bits(),
            });
        }
    }
    Ok(vk::Viewport {
        x: x as f32,
        // The bottom edge, because the height below is negative and a viewport
        // is measured from its `y`.
        y: (y + height) as f32,
        width: width as f32,
        height: -(height as f32),
        min_depth: near as f32,
        max_depth: far as f32,
    })
}

/// The scissor a guest rectangle becomes.
///
/// No flip: a scissor is in framebuffer pixels in both models, with the origin
/// at the top left. Flipping it as well would move the rectangle off the
/// geometry the viewport just placed.
///
/// # Errors
///
/// [`Refusal::OutOfRange`] for a coordinate outside the 32-bit fields Vulkan
/// carries.
pub fn scissor(guest: ScissorRect) -> Result<vk::Rect2D, Refusal> {
    let offset_x = i32::try_from(guest.x).map_err(|_| Refusal::OutOfRange {
        field: "x",
        value: guest.x,
    })?;
    let offset_y = i32::try_from(guest.y).map_err(|_| Refusal::OutOfRange {
        field: "y",
        value: guest.y,
    })?;
    let width = u32::try_from(guest.width).map_err(|_| Refusal::OutOfRange {
        field: "width",
        value: guest.width,
    })?;
    let height = u32::try_from(guest.height).map_err(|_| Refusal::OutOfRange {
        field: "height",
        value: guest.height,
    })?;
    Ok(vk::Rect2D {
        offset: vk::Offset2D {
            x: offset_x,
            y: offset_y,
        },
        extent: vk::Extent2D { width, height },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::render::FloatBits;
    use std::collections::BTreeSet;

    /// Every capability, and every state dynamic.
    fn all() -> RasterCell {
        RasterCell {
            depth_clamp: true,
            fill_mode_non_solid: true,
            dynamic_cull_and_winding: true,
            dynamic_polygon_mode: true,
            dynamic_depth_clamp: true,
        }
    }

    /// Every capability, and nothing dynamic — the host that bakes all four
    /// states into the pipeline.
    fn baked() -> RasterCell {
        RasterCell {
            dynamic_cull_and_winding: false,
            dynamic_polygon_mode: false,
            dynamic_depth_clamp: false,
            ..all()
        }
    }

    /// A guest that set all four states away from their defaults, so a plan
    /// that dropped one on the floor is visible rather than accidentally
    /// right.
    const ALL_SET: GuestRasterState = GuestRasterState {
        cull_mode: MTL_CULL_MODE_BACK,
        winding: MTL_WINDING_COUNTER_CLOCKWISE,
        depth_clip_mode: MTL_DEPTH_CLIP_MODE_CLAMP,
        fill_mode: MTL_TRIANGLE_FILL_MODE_LINES,
    };

    fn guest_viewport(x: f64, y: f64, width: f64, height: f64) -> Viewport {
        Viewport {
            origin_x_bits: x.to_bits(),
            origin_y_bits: y.to_bits(),
            width_bits: width.to_bits(),
            height_bits: height.to_bits(),
            z_near_bits: 0.0f64.to_bits(),
            z_far_bits: 1.0f64.to_bits(),
        }
    }

    #[test]
    fn each_cull_ordinal_is_its_own_face_and_nothing_else_is_a_face() {
        assert_eq!(cull_mode(MTL_CULL_MODE_NONE), Ok(vk::CullModeFlags::NONE));
        assert_eq!(cull_mode(MTL_CULL_MODE_FRONT), Ok(vk::CullModeFlags::FRONT));
        assert_eq!(cull_mode(MTL_CULL_MODE_BACK), Ok(vk::CullModeFlags::BACK));
        for ordinal in [3u64, 4, u64::MAX] {
            // Not folded onto `NONE`, which would draw the back faces the
            // guest culled.
            assert_eq!(
                cull_mode(ordinal),
                Err(Refusal::UnknownOrdinal {
                    state: "cull_mode",
                    ordinal,
                })
            );
        }
    }

    #[test]
    fn each_winding_is_its_own_and_nothing_else_is_a_winding() {
        assert_eq!(
            front_face(MTL_WINDING_CLOCKWISE),
            Ok(vk::FrontFace::CLOCKWISE)
        );
        assert_eq!(
            front_face(MTL_WINDING_COUNTER_CLOCKWISE),
            Ok(vk::FrontFace::COUNTER_CLOCKWISE)
        );
        assert!(front_face(2).is_err());
    }

    #[test]
    fn the_default_of_each_capability_state_needs_no_capability() {
        let none = RasterCell::default();
        // A host with neither feature still runs every guest that does not ask
        // for one.
        assert_eq!(depth_clamp(MTL_DEPTH_CLIP_MODE_CLIP, none), Ok(false));
        assert_eq!(
            polygon_mode(MTL_TRIANGLE_FILL_MODE_FILL, none),
            Ok(vk::PolygonMode::FILL)
        );
    }

    #[test]
    fn clamping_and_wireframe_refuse_where_the_feature_is_absent() {
        let none = RasterCell::default();
        assert_eq!(
            depth_clamp(MTL_DEPTH_CLIP_MODE_CLAMP, none),
            Err(Refusal::NoDepthClamp)
        );
        assert_eq!(
            polygon_mode(MTL_TRIANGLE_FILL_MODE_LINES, none),
            Err(Refusal::NoNonSolidFill)
        );
        // And are honoured where it is present, so neither refusal is
        // unconditional.
        assert_eq!(depth_clamp(MTL_DEPTH_CLIP_MODE_CLAMP, all()), Ok(true));
        assert_eq!(
            polygon_mode(MTL_TRIANGLE_FILL_MODE_LINES, all()),
            Ok(vk::PolygonMode::LINE)
        );
    }

    #[test]
    fn an_unknown_ordinal_refuses_as_that_and_not_as_a_missing_capability() {
        // The distinction matters: one is a decode or contract gap and the
        // other is a host that cannot do what the guest asked.
        assert!(matches!(
            depth_clamp(7, all()),
            Err(Refusal::UnknownOrdinal { .. })
        ));
        assert!(matches!(
            polygon_mode(7, all()),
            Err(Refusal::UnknownOrdinal { .. })
        ));
        // And on a host with neither feature, an unknown ordinal is still an
        // unknown ordinal rather than the capability refusal.
        assert!(matches!(
            polygon_mode(7, RasterCell::default()),
            Err(Refusal::UnknownOrdinal { .. })
        ));
    }

    #[test]
    fn the_viewport_flips_y_and_the_bottom_edge_is_where_it_starts() {
        // A 800x600 viewport at the origin. Vulkan measures from `y` with a
        // negative height, so `y` is the bottom edge and the covered band is
        // still 0..600.
        let flipped = viewport(guest_viewport(0.0, 0.0, 800.0, 600.0)).expect("finite");
        assert!((flipped.x - 0.0).abs() < f32::EPSILON);
        assert!((flipped.y - 600.0).abs() < f32::EPSILON);
        assert!((flipped.width - 800.0).abs() < f32::EPSILON);
        assert!((flipped.height + 600.0).abs() < f32::EPSILON);
        assert!(flipped.height < 0.0, "the flip is the negative height");
        // The band the viewport covers is unchanged; only its direction is
        // reversed.
        assert!((flipped.y + flipped.height - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn an_offset_viewport_keeps_its_band_after_the_flip() {
        // 100..340 vertically, which must still be 100..340 after the flip.
        let flipped = viewport(guest_viewport(50.0, 100.0, 200.0, 240.0)).expect("finite");
        assert!((flipped.y - 340.0).abs() < f32::EPSILON);
        assert!((flipped.y + flipped.height - 100.0).abs() < f32::EPSILON);
        assert!((flipped.x - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn the_depth_range_is_carried_and_not_flipped() {
        // Metal and Vulkan both use [0, 1] depth, so this is a narrowing and
        // never a remap. A guest that reversed its own range keeps it.
        let mut guest = guest_viewport(0.0, 0.0, 1.0, 1.0);
        guest.z_near_bits = 1.0f64.to_bits();
        guest.z_far_bits = 0.0f64.to_bits();
        let flipped = viewport(guest).expect("finite");
        assert!((flipped.min_depth - 1.0).abs() < f32::EPSILON);
        assert!((flipped.max_depth - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn a_viewport_that_is_not_a_number_refuses_rather_than_rasterizing() {
        for bits in [f64::NAN.to_bits(), f64::INFINITY.to_bits()] {
            let mut guest = guest_viewport(0.0, 0.0, 1.0, 1.0);
            guest.width_bits = bits;
            assert!(matches!(
                viewport(guest),
                Err(Refusal::OutOfRange { field: "width", .. })
            ));
        }
    }

    #[test]
    fn a_scissor_is_not_flipped() {
        let rect = scissor(ScissorRect {
            x: 10,
            y: 20,
            width: 30,
            height: 40,
        })
        .expect("in range");
        // Same origin, same corner: flipping this as well would move the
        // rectangle off the geometry the viewport just placed.
        assert_eq!(rect.offset, vk::Offset2D { x: 10, y: 20 });
        assert_eq!(
            rect.extent,
            vk::Extent2D {
                width: 30,
                height: 40
            }
        );
    }

    #[test]
    fn a_scissor_outside_the_native_fields_refuses_rather_than_truncating() {
        assert_eq!(
            scissor(ScissorRect {
                x: u64::from(u32::MAX) + 1,
                y: 0,
                width: 1,
                height: 1,
            }),
            Err(Refusal::OutOfRange {
                field: "x",
                value: u64::from(u32::MAX) + 1,
            })
        );
        assert!(matches!(
            scissor(ScissorRect {
                x: 0,
                y: 0,
                width: u64::from(u32::MAX) + 1,
                height: 1,
            }),
            Err(Refusal::OutOfRange { field: "width", .. })
        ));
    }

    #[test]
    fn float_state_narrows_without_reinterpretation() {
        // The bit-pattern types exist so a state table can compare; they are
        // not a wire encoding to be passed through.
        assert!((FloatBits::from_f32(0.25).to_f32() - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn every_refusal_names_itself() {
        let refusals = [
            Refusal::UnknownOrdinal {
                state: "cull_mode",
                ordinal: 9,
            },
            Refusal::NoDepthClamp,
            Refusal::NoNonSolidFill,
            Refusal::OutOfRange {
                field: "x",
                value: 1,
            },
        ];
        let slugs: BTreeSet<&str> = refusals.iter().map(|r| r.slug()).collect();
        assert_eq!(slugs.len(), refusals.len());
        for refusal in refusals {
            assert!(refusal.to_string().starts_with(refusal.slug()));
            assert!(refusal.slug().starts_with("vk_raster_"));
        }
    }

    /// Nothing dynamic: every state the guest set lands in the pipeline, and
    /// the encoder is asked for none of them.
    #[test]
    fn a_host_with_no_dynamic_state_bakes_all_four_states_into_the_pipeline() {
        let plan = plan(ALL_SET, baked()).expect("every capability is present");
        assert_eq!(plan.state.cull_mode, vk::CullModeFlags::BACK);
        assert_eq!(plan.state.front_face, vk::FrontFace::COUNTER_CLOCKWISE);
        assert_eq!(plan.state.polygon_mode, vk::PolygonMode::LINE);
        assert!(plan.state.depth_clamp_enable);
        assert_eq!(plan.dynamic, DynamicRaster::default());
        // Only the unconditional one, which is depth bias.
        assert_eq!(
            plan.state.dynamic.states(),
            vec![vk::DynamicState::DEPTH_BIAS]
        );
    }

    /// Every state dynamic: the guest's values move to the encoder and the
    /// pipeline holds the default in each place.
    ///
    /// This is the whole point of the cell. If the guest's value stayed in the
    /// pipeline state as well, it would still be a cache dimension and the
    /// dynamic state would have bought nothing — so the assertion is that the
    /// pipeline members are *not* what the guest asked for.
    #[test]
    fn a_dynamic_state_leaves_the_pipeline_holding_the_default_and_not_the_guests_value() {
        let plan = plan(ALL_SET, all()).expect("every capability is present");
        assert_eq!(plan.state.cull_mode, vk::CullModeFlags::NONE);
        assert_eq!(plan.state.front_face, vk::FrontFace::CLOCKWISE);
        assert_eq!(plan.state.polygon_mode, vk::PolygonMode::FILL);
        assert!(!plan.state.depth_clamp_enable);
        assert_eq!(
            plan.dynamic,
            DynamicRaster {
                cull_mode: Some(vk::CullModeFlags::BACK),
                front_face: Some(vk::FrontFace::COUNTER_CLOCKWISE),
                polygon_mode: Some(vk::PolygonMode::LINE),
                depth_clamp_enable: Some(true),
            }
        );
    }

    /// The one claim a cache rests on: on a fully dynamic host, every guest
    /// rasterizer state plans to one pipeline state, so the whole sixteen-way
    /// product compiles once.
    #[test]
    fn every_guest_state_collapses_to_one_pipeline_where_all_four_are_dynamic() {
        let mut states = BTreeSet::new();
        let mut baked_states = BTreeSet::new();
        let mut count = 0;
        for cull_mode in [MTL_CULL_MODE_NONE, MTL_CULL_MODE_FRONT, MTL_CULL_MODE_BACK] {
            for winding in [MTL_WINDING_CLOCKWISE, MTL_WINDING_COUNTER_CLOCKWISE] {
                for depth_clip_mode in [MTL_DEPTH_CLIP_MODE_CLIP, MTL_DEPTH_CLIP_MODE_CLAMP] {
                    for fill_mode in [MTL_TRIANGLE_FILL_MODE_FILL, MTL_TRIANGLE_FILL_MODE_LINES] {
                        let guest = GuestRasterState {
                            cull_mode,
                            winding,
                            depth_clip_mode,
                            fill_mode,
                        };
                        let dynamic = plan(guest, all()).expect("capable host");
                        states.insert(format!("{:?}", dynamic.state));
                        // The same guest states against a host that bakes
                        // them. This is what makes the count above a saving
                        // rather than a plan that forgot to read the guest:
                        // the two hosts must disagree about how many pipelines
                        // this product needs.
                        let baked = plan(guest, baked()).expect("capable host");
                        baked_states.insert(format!("{:?}", baked.state));
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, 24, "the whole product was walked");
        // Injective the other way: baking gives a pipeline per state, and the
        // three cull modes, two windings, two clip modes and two fill modes
        // are all distinguishable. Twenty-four compilations against one.
        assert_eq!(baked_states.len(), 24, "baking collapses nothing");
        assert_eq!(
            states.len(),
            1,
            "one pipeline serves every rasterizer state"
        );
    }

    /// A dynamic state still needs its feature. `vkCmdSetPolygonModeEXT` with
    /// `LINE` is as invalid without `fillModeNonSolid` as the pipeline member
    /// is, so moving the state to the encoder must not move the refusal.
    #[test]
    fn making_a_state_dynamic_does_not_make_its_capability_free() {
        // A host that offers the dynamic member and not the feature it needs.
        // Contrived, and exactly the host the refusal exists for.
        let cell = RasterCell {
            depth_clamp: false,
            fill_mode_non_solid: false,
            dynamic_cull_and_winding: true,
            dynamic_polygon_mode: true,
            dynamic_depth_clamp: true,
        };
        assert_eq!(
            plan(
                GuestRasterState {
                    fill_mode: MTL_TRIANGLE_FILL_MODE_LINES,
                    ..GuestRasterState::DEFAULT
                },
                cell
            ),
            Err(Refusal::NoNonSolidFill)
        );
        assert_eq!(
            plan(
                GuestRasterState {
                    depth_clip_mode: MTL_DEPTH_CLIP_MODE_CLAMP,
                    ..GuestRasterState::DEFAULT
                },
                cell
            ),
            Err(Refusal::NoDepthClamp)
        );
        // And the default of each still runs on it, dynamic or not.
        assert!(plan(GuestRasterState::DEFAULT, cell).is_ok());
    }

    /// An unrecognised ordinal refuses whether or not its state is dynamic.
    /// The parse is not skipped because the value is going to a command.
    #[test]
    fn an_unknown_ordinal_refuses_on_the_dynamic_path_too() {
        for cell in [all(), baked()] {
            assert!(plan(
                GuestRasterState {
                    cull_mode: 7,
                    ..GuestRasterState::DEFAULT
                },
                cell
            )
            .is_err());
            assert!(plan(
                GuestRasterState {
                    winding: 7,
                    ..GuestRasterState::DEFAULT
                },
                cell
            )
            .is_err());
        }
    }

    /// The two halves cannot disagree: a member is a placeholder in the
    /// pipeline state exactly where the encoder is asked to supply it.
    #[test]
    fn a_member_is_dynamic_in_exactly_one_of_the_two_halves() {
        // Each of the eight subsets a host can offer, against a guest that set
        // every state away from its default.
        for bits in 0u8..8 {
            let cell = RasterCell {
                depth_clamp: true,
                fill_mode_non_solid: true,
                dynamic_cull_and_winding: bits & 1 != 0,
                dynamic_polygon_mode: bits & 2 != 0,
                dynamic_depth_clamp: bits & 4 != 0,
            };
            let p = plan(ALL_SET, cell).expect("every capability is present");
            let d = p.state.dynamic;
            assert_eq!(d, RasterDynamic::of(cell));
            assert_eq!(p.dynamic.cull_mode.is_some(), d.cull_and_winding);
            assert_eq!(p.dynamic.front_face.is_some(), d.cull_and_winding);
            assert_eq!(p.dynamic.polygon_mode.is_some(), d.polygon_mode);
            assert_eq!(p.dynamic.depth_clamp_enable.is_some(), d.depth_clamp_enable);
            // The placeholder is the default, and the guest asked for
            // something else, so "dynamic" and "baked" are visibly different
            // in the pipeline state as well.
            assert_eq!(
                p.state.cull_mode == vk::CullModeFlags::NONE,
                d.cull_and_winding
            );
            assert_eq!(
                p.state.polygon_mode == vk::PolygonMode::FILL,
                d.polygon_mode
            );
            assert_eq!(!p.state.depth_clamp_enable, d.depth_clamp_enable);
        }
    }

    /// The dynamic-state list has no duplicates and grows only with the cell.
    #[test]
    fn the_dynamic_state_list_names_each_member_once_and_depth_bias_always() {
        for bits in 0u8..8 {
            let dynamic = RasterDynamic {
                cull_and_winding: bits & 1 != 0,
                polygon_mode: bits & 2 != 0,
                depth_clamp_enable: bits & 4 != 0,
            };
            let states = dynamic.states();
            let unique: BTreeSet<_> = states.iter().map(|s| s.as_raw()).collect();
            assert_eq!(unique.len(), states.len(), "{dynamic:?} repeats a state");
            assert!(states.contains(&vk::DynamicState::DEPTH_BIAS));
            // One entry for depth bias, two for the pair, one each for the
            // other two.
            let expected = 1
                + usize::from(dynamic.cull_and_winding) * 2
                + usize::from(dynamic.polygon_mode)
                + usize::from(dynamic.depth_clamp_enable);
            assert_eq!(states.len(), expected);
        }
        assert_eq!(
            RasterDynamic::NONE.states(),
            vec![vk::DynamicState::DEPTH_BIAS]
        );
    }

    /// Depth bias is enabled on every pipeline this rail builds, and its three
    /// values start at zero. Metal has no pipeline-level enable, so a clear
    /// one would silently drop `setDepthBias:` — see the module doc.
    #[test]
    fn every_pipeline_permits_a_depth_bias_and_starts_at_none() {
        for cell in [all(), baked(), RasterCell::default()] {
            let p = plan(GuestRasterState::DEFAULT, cell).expect("the defaults need no feature");
            assert!(p.state.depth_bias_enable);
            let native = p.state.native();
            assert_eq!(native.depth_bias_enable, vk::TRUE);
            assert_eq!(native.depth_bias_constant_factor, 0.0);
            assert_eq!(native.depth_bias_clamp, 0.0);
            assert_eq!(native.depth_bias_slope_factor, 0.0);
            // Metal has neither, and both would drop or widen geometry.
            assert_eq!(native.rasterizer_discard_enable, vk::FALSE);
            assert_eq!(native.line_width, 1.0);
        }
    }

    /// The native structure carries what the plan says, field for field.
    #[test]
    fn the_native_structure_is_the_plan_and_not_the_guests_state() {
        let p = plan(ALL_SET, baked()).expect("every capability is present");
        let native = p.state.native();
        assert_eq!(native.cull_mode, vk::CullModeFlags::BACK);
        assert_eq!(native.front_face, vk::FrontFace::COUNTER_CLOCKWISE);
        assert_eq!(native.polygon_mode, vk::PolygonMode::LINE);
        assert_eq!(native.depth_clamp_enable, vk::TRUE);

        let dyn_ = plan(ALL_SET, all()).expect("every capability is present");
        let native = dyn_.state.native();
        assert_eq!(native.cull_mode, vk::CullModeFlags::NONE);
        assert_eq!(native.depth_clamp_enable, vk::FALSE);
    }
}
