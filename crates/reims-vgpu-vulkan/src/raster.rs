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

    fn all() -> RasterCell {
        RasterCell {
            depth_clamp: true,
            fill_mode_non_solid: true,
        }
    }

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
}
