//! Generating a texture's mip chain: the blit ladder, and the layout each rung
//! has to be in before the next one reads it.
//!
//! # It is a ladder and the order is the whole content
//!
//! Level *n* is produced by filtering level *n-1*, so level *n-1* has to be a
//! finished transfer *destination* before it can be a transfer *source*, and
//! the transition between those two facts is the barrier that makes the chain
//! correct rather than racy. A generator that transitioned the whole image
//! once and then issued N blits would read level *n-1* while the blit
//! producing it was still running, and the result is a mip chain that is
//! usually right and sometimes not — the shape of bug that survives a test
//! suite.
//!
//! So the plan is a sequence of steps, alternating, and the sequence *is* the
//! claim this module makes.
//!
//! # A filtered reduction needs a filter
//!
//! `generateMipmapsForTexture:` filters. `vkCmdBlitImage` with
//! `VK_FILTER_LINEAR` needs the format to report
//! `SAMPLED_IMAGE_FILTER_LINEAR` for optimal tiling, and not every format
//! does. Falling back to `VK_FILTER_NEAREST` would produce a chain that
//! allocates, records and runs, and looks wrong — an aliased, sparkling
//! texture the guest has no way to attribute. So a format without the
//! capability refuses by name, and the capability is *measured* off the
//! physical device rather than assumed from the format.
//!
//! Depth and stencil formats refuse for a different reason: linear filtering
//! of a depth image is invalid usage regardless of what the format reports,
//! because there is no meaningful average of two depth values.
//!
//! # One level is a no-op, not a refusal
//!
//! A texture with a single level has nothing to generate, and the guest's own
//! call on one does nothing. An empty plan says exactly that, and a refusal
//! there would turn a legal guest command into a lost frame.
//!
//! # Planned, not recorded
//!
//! The steps are values. The layout tracker is consulted, and it is consulted
//! only after every refusal has been decided — so a refused generation leaves
//! the tracker exactly as it was, and a caller that retries is not fighting
//! half-applied state.

use ash::vk;
use reims_vgpu_core::pixel_format::{format_has_depth_aspect, format_has_stencil_aspect};
use reims_vgpu_core::texture_shape::Texture;

use crate::layout::{Contents, Decline, ImageId, LayoutTracker, Subresource, Transition, Use};

/// What this host reports about filtering one format, measured rather than
/// assumed.
///
/// One bool because one question is asked. It comes from
/// `VkFormatProperties::optimalTilingFeatures &
/// VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT`, which is the bit
/// `vkCmdBlitImage` with a linear filter requires of its *source*.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FilterSupport {
    pub linear_blit_source: bool,
}

/// Why a mip chain cannot be generated on this host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The format cannot be linearly filtered here. Named rather than dropped
    /// to nearest, which would produce a chain that runs and looks wrong.
    NoLinearFilter { format: u16 },
    /// Depth and stencil have no filtered reduction on any host.
    DepthStencil { format: u16 },
    /// A multisample texture has no mip chain to generate and no blit that
    /// could produce one.
    Multisampled { samples: u32 },
    /// The layout tracker does not know this image or subresource.
    Untracked { decline: Decline },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NoLinearFilter { .. } => "vk_mipmap_no_linear_filter",
            Self::DepthStencil { .. } => "vk_mipmap_depth_stencil",
            Self::Multisampled { .. } => "vk_mipmap_multisampled",
            Self::Untracked { .. } => "vk_mipmap_untracked",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoLinearFilter { format } | Self::DepthStencil { format } => {
                write!(f, "{} format={format}", self.slug())
            }
            Self::Multisampled { samples } => write!(f, "{} samples={samples}", self.slug()),
            Self::Untracked { decline } => write!(f, "{} {decline}", self.slug()),
        }
    }
}

/// One blit rung, in the fields `VkImageBlit` takes.
///
/// Spelled out because ash's is neither comparable nor `Eq`, and a ladder
/// whose rungs cannot be compared is one whose order cannot be asserted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rung {
    pub source_level: u32,
    pub dest_level: u32,
    pub layers: u32,
    pub aspect: vk::ImageAspectFlags,
    /// The source level's extent, and the destination's, in that order.
    pub source_extent: vk::Extent3D,
    pub dest_extent: vk::Extent3D,
}

impl Rung {
    pub fn native(self) -> vk::ImageBlit {
        let corner = |extent: vk::Extent3D| {
            [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: extent.width as i32,
                    y: extent.height as i32,
                    z: extent.depth as i32,
                },
            ]
        };
        vk::ImageBlit {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: self.aspect,
                mip_level: self.source_level,
                base_array_layer: 0,
                layer_count: self.layers,
            },
            src_offsets: corner(self.source_extent),
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: self.aspect,
                mip_level: self.dest_level,
                base_array_layer: 0,
                layer_count: self.layers,
            },
            dst_offsets: corner(self.dest_extent),
        }
    }
}

/// One step of the ladder, in the order it must be recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// A layout transition of one subresource. `None` never appears here: a
    /// step that changes nothing is not emitted.
    Transition(Transition),
    /// The filtered reduction of one level into the next.
    Blit(Rung),
}

/// The steps that build `texture`'s mip chain from its top level.
///
/// Empty when there is nothing to generate.
///
/// # Errors
///
/// [`Refusal`], decided in full before the tracker is touched — so a refused
/// generation leaves the tracker exactly as it was.
pub fn plan(
    image: ImageId,
    texture: Texture,
    support: FilterSupport,
    tracker: &mut LayoutTracker,
) -> Result<Vec<Step>, Refusal> {
    let format = texture.pixel_format();
    if texture.sample_count() > 1 {
        return Err(Refusal::Multisampled {
            samples: texture.sample_count(),
        });
    }
    if format_has_depth_aspect(format) || format_has_stencil_aspect(format) {
        return Err(Refusal::DepthStencil { format });
    }
    if !support.linear_blit_source {
        return Err(Refusal::NoLinearFilter { format });
    }
    if texture.mip_levels() < 2 {
        return Ok(Vec::new());
    }

    let layers = texture.layers();
    let aspect = crate::view::aspect(format);
    let extent = |level: u32| {
        let level_extent = texture
            .level_extent(level)
            .expect("levels below the declared count exist");
        vk::Extent3D {
            width: level_extent.x,
            height: level_extent.y,
            depth: level_extent.z,
        }
    };

    // Every transition this ladder needs, resolved before any of them is
    // applied. A refusal from the tracker halfway through would leave some
    // subresources moved and some not, and there is no way back from that.
    let mut needed: Vec<(Subresource, Use, Contents)> = Vec::new();
    for level in 1..texture.mip_levels() {
        for layer in 0..layers {
            // The source's contents are the point, so `Keep`. The
            // destination's are about to be entirely overwritten by the blit,
            // so `Discard` — which is the one case where throwing bytes away
            // is a saving rather than a loss.
            needed.push((
                Subresource::new(level - 1, layer),
                Use::TransferSrc,
                Contents::Keep,
            ));
            needed.push((
                Subresource::new(level, layer),
                Use::TransferDst,
                Contents::Discard,
            ));
        }
    }
    for (subresource, _, _) in &needed {
        tracker
            .layout(image, *subresource)
            .map_err(|decline| Refusal::Untracked { decline })?;
    }

    let mut steps = Vec::new();
    let mut pending = needed.into_iter();
    for level in 1..texture.mip_levels() {
        for _ in 0..layers {
            for _ in 0..2 {
                let (subresource, use_, contents) =
                    pending.next().expect("one entry per emitted step");
                if let Some(transition) = tracker
                    .plan(image, subresource, use_, contents)
                    .map_err(|decline| Refusal::Untracked { decline })?
                {
                    steps.push(Step::Transition(transition));
                }
            }
        }
        steps.push(Step::Blit(Rung {
            source_level: level - 1,
            dest_level: level,
            layers,
            aspect,
            source_extent: extent(level - 1),
            dest_extent: extent(level),
        }));
    }
    Ok(steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::pixel_format::{MTL_FORMAT_DEPTH32_FLOAT, MTL_FORMAT_RGBA8_UNORM};
    use reims_vgpu_core::texture_shape::{TextureKind, TextureShape, TextureUsage};
    use std::collections::BTreeSet;

    const IMAGE: ImageId = ImageId(1);

    fn filterable() -> FilterSupport {
        FilterSupport {
            linear_blit_source: true,
        }
    }

    fn texture(kind: TextureKind, width: u32, levels: u32, layers: u32, format: u16) -> Texture {
        TextureShape {
            kind: kind.ordinal(),
            width,
            height: width,
            depth: if kind.dimensions() == reims_vgpu_core::texture_shape::Dimensions::Three {
                width
            } else {
                1
            },
            mipmap_level_count: levels,
            sample_count: if kind.is_multisample() { 4 } else { 1 },
            array_length: layers,
            pixel_format: format,
            usage: TextureUsage::SHADER_READ,
        }
        .checked()
        .expect("a valid declaration")
    }

    fn tracked(texture: Texture) -> LayoutTracker {
        let mut tracker = LayoutTracker::new();
        tracker.declare(IMAGE, texture.mip_levels(), texture.layers(), 1, Some(0));
        tracker
    }

    fn flat() -> Texture {
        texture(TextureKind::D2, 16, 5, 1, MTL_FORMAT_RGBA8_UNORM)
    }

    #[test]
    fn the_ladder_alternates_transitions_and_blits_in_order() {
        let texture = flat();
        let mut tracker = tracked(texture);
        let steps = plan(IMAGE, texture, filterable(), &mut tracker).expect("plannable");

        // Four rungs for five levels, each preceded by exactly the two
        // transitions that make the level below readable and this one
        // writable.
        let blits: Vec<&Rung> = steps
            .iter()
            .filter_map(|s| match s {
                Step::Blit(rung) => Some(rung),
                Step::Transition(_) => None,
            })
            .collect();
        assert_eq!(blits.len(), 4);
        for (index, rung) in blits.iter().enumerate() {
            assert_eq!(rung.source_level, index as u32);
            assert_eq!(rung.dest_level, index as u32 + 1);
        }

        // And the order: every blit's source has been moved to TRANSFER_SRC
        // and its destination to TRANSFER_DST before it appears.
        let mut src = Vec::new();
        let mut dst = Vec::new();
        for step in &steps {
            match step {
                Step::Transition(t) => {
                    if t.to == vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
                        src.push(t.subresource.level);
                    } else {
                        dst.push(t.subresource.level);
                    }
                }
                Step::Blit(rung) => {
                    assert!(
                        src.contains(&rung.source_level),
                        "level {} blitted before it became a source",
                        rung.source_level
                    );
                    assert!(
                        dst.contains(&rung.dest_level),
                        "level {} written before it became a destination",
                        rung.dest_level
                    );
                }
            }
        }
    }

    #[test]
    fn each_rung_halves_the_extent_and_the_last_one_reaches_a_single_texel() {
        let texture = flat();
        let mut tracker = tracked(texture);
        let steps = plan(IMAGE, texture, filterable(), &mut tracker).expect("plannable");
        let extents: Vec<(u32, u32)> = steps
            .iter()
            .filter_map(|s| match s {
                Step::Blit(rung) => Some((rung.source_extent.width, rung.dest_extent.width)),
                Step::Transition(_) => None,
            })
            .collect();
        assert_eq!(extents, [(16, 8), (8, 4), (4, 2), (2, 1)]);
    }

    #[test]
    fn a_volume_reduces_its_depth_along_with_the_rest() {
        let texture = texture(TextureKind::D3, 8, 4, 1, MTL_FORMAT_RGBA8_UNORM);
        let mut tracker = tracked(texture);
        let steps = plan(IMAGE, texture, filterable(), &mut tracker).expect("plannable");
        let depths: Vec<(u32, u32)> = steps
            .iter()
            .filter_map(|s| match s {
                Step::Blit(rung) => Some((rung.source_extent.depth, rung.dest_extent.depth)),
                Step::Transition(_) => None,
            })
            .collect();
        // A 2D texture's depth stays one at every level; a volume's does not,
        // and a ladder that copied the base depth would read past every level
        // below the first.
        assert_eq!(depths, [(8, 4), (4, 2), (2, 1)]);
    }

    #[test]
    fn every_layer_of_an_array_is_moved_and_the_blit_covers_them_at_once() {
        let texture = texture(TextureKind::D2Array, 8, 3, 4, MTL_FORMAT_RGBA8_UNORM);
        let mut tracker = tracked(texture);
        let steps = plan(IMAGE, texture, filterable(), &mut tracker).expect("plannable");

        // One transition per (level, layer) pair the ladder touches: layouts
        // are per subresource, so a per-level transition would leave three of
        // four layers in the wrong one.
        let moved: BTreeSet<(u32, u32)> = steps
            .iter()
            .filter_map(|s| match s {
                Step::Transition(t) => Some((t.subresource.level, t.subresource.layer)),
                Step::Blit(_) => None,
            })
            .collect();
        assert_eq!(moved.len(), 3 * 4);

        // The blit itself covers the whole layer span in one region.
        for step in &steps {
            if let Step::Blit(rung) = step {
                assert_eq!(rung.layers, 4);
            }
        }
    }

    #[test]
    fn the_destination_of_each_rung_discards_and_the_source_keeps() {
        let texture = flat();
        let mut tracker = tracked(texture);
        let steps = plan(IMAGE, texture, filterable(), &mut tracker).expect("plannable");
        for step in &steps {
            if let Step::Transition(t) = step {
                if t.to == vk::ImageLayout::TRANSFER_DST_OPTIMAL {
                    // Every texel is about to be overwritten, so preserving
                    // them is bandwidth spent on bytes nobody reads.
                    assert!(t.discarded_contents, "level {}", t.subresource.level);
                    assert_eq!(t.from, vk::ImageLayout::UNDEFINED);
                } else {
                    // The source's contents are the entire point.
                    assert!(!t.discarded_contents, "level {}", t.subresource.level);
                }
            }
        }
    }

    #[test]
    fn a_single_level_texture_generates_nothing_rather_than_refusing() {
        let texture = texture(TextureKind::D2, 16, 1, 1, MTL_FORMAT_RGBA8_UNORM);
        let mut tracker = tracked(texture);
        assert_eq!(
            plan(IMAGE, texture, filterable(), &mut tracker),
            Ok(Vec::new())
        );
        assert_eq!(tracker.census().transitions, 0);
    }

    #[test]
    fn a_format_this_host_cannot_filter_refuses_rather_than_dropping_to_nearest() {
        let texture = flat();
        let mut tracker = tracked(texture);
        assert_eq!(
            plan(IMAGE, texture, FilterSupport::default(), &mut tracker),
            Err(Refusal::NoLinearFilter {
                format: MTL_FORMAT_RGBA8_UNORM
            })
        );
        // Nothing was moved: a refused generation is retryable against the
        // same tracker.
        assert_eq!(tracker.census().transitions, 0);
    }

    #[test]
    fn a_depth_texture_refuses_even_when_the_host_reports_filtering() {
        let texture = texture(TextureKind::D2, 16, 5, 1, MTL_FORMAT_DEPTH32_FLOAT);
        let mut tracker = tracked(texture);
        // Reported filterable and still refused: there is no meaningful
        // average of two depth values, so the capability is not the question.
        assert_eq!(
            plan(IMAGE, texture, filterable(), &mut tracker),
            Err(Refusal::DepthStencil {
                format: MTL_FORMAT_DEPTH32_FLOAT
            })
        );
        assert_eq!(tracker.census().transitions, 0);
    }

    #[test]
    fn a_multisample_texture_refuses_before_anything_else_is_asked() {
        let texture = texture(
            TextureKind::D2Multisample,
            16,
            1,
            1,
            MTL_FORMAT_DEPTH32_FLOAT,
        );
        let mut tracker = tracked(texture);
        assert_eq!(
            plan(IMAGE, texture, FilterSupport::default(), &mut tracker),
            Err(Refusal::Multisampled { samples: 4 })
        );
    }

    #[test]
    fn an_untracked_image_refuses_without_moving_anything() {
        let texture = flat();
        let mut tracker = LayoutTracker::new();
        let refusal =
            plan(IMAGE, texture, filterable(), &mut tracker).expect_err("nothing was declared");
        assert!(matches!(refusal, Refusal::Untracked { .. }));
        assert_eq!(tracker.census().transitions, 0);
    }

    #[test]
    fn a_rung_becomes_a_blit_whose_corners_are_the_two_level_extents() {
        let rung = Rung {
            source_level: 1,
            dest_level: 2,
            layers: 3,
            aspect: vk::ImageAspectFlags::COLOR,
            source_extent: vk::Extent3D {
                width: 8,
                height: 4,
                depth: 1,
            },
            dest_extent: vk::Extent3D {
                width: 4,
                height: 2,
                depth: 1,
            },
        };
        let blit = rung.native();
        assert_eq!(blit.src_subresource.mip_level, 1);
        assert_eq!(blit.dst_subresource.mip_level, 2);
        assert_eq!(blit.src_subresource.layer_count, 3);
        assert_eq!(blit.src_offsets[0].x, 0);
        assert_eq!(blit.src_offsets[1].x, 8);
        assert_eq!(blit.src_offsets[1].y, 4);
        assert_eq!(blit.dst_offsets[1].x, 4);
        assert_eq!(blit.dst_offsets[1].z, 1);
    }

    #[test]
    fn every_refusal_names_itself() {
        let refusals = [
            Refusal::NoLinearFilter { format: 1 },
            Refusal::DepthStencil { format: 1 },
            Refusal::Multisampled { samples: 4 },
            Refusal::Untracked {
                decline: Decline::UnknownImage { image: IMAGE },
            },
        ];
        let slugs: BTreeSet<&str> = refusals.iter().map(|r| r.slug()).collect();
        assert_eq!(slugs.len(), refusals.len());
        for refusal in refusals {
            assert!(refusal.to_string().starts_with(refusal.slug()));
            assert!(refusal.slug().starts_with("vk_mipmap_"));
        }
    }
}
