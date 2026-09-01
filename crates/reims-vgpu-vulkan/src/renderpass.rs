//! What both rungs of [`crate::pass::Carrier`] need built, and the two
//! different keys the objects they build are cached under.
//!
//! # A pipeline is compatible with a pass it was never built for
//!
//! Vulkan's render-pass compatibility rules say two passes are compatible when
//! their attachments agree on *format* and *sample count*. Load and store
//! operations and image layouts are explicitly not part of it. So a pipeline
//! built against a pass that clears is usable in one that loads, and the two
//! are different `VkRenderPass` objects.
//!
//! That is why there are two keys here and not one. [`Compatibility`] is what
//! a pipeline cache is keyed on; [`Signature`] is what a `VkRenderPass` cache
//! is keyed on, and it *contains* a `Compatibility` rather than restating it,
//! so the two cannot disagree about a format. Keying pipelines on the
//! signature would be correct and would recompile every pipeline in a frame
//! the first time a guest changed a load action from clear to load — which
//! guests do between the first pass of a frame and the rest of it.
//!
//! # The same key on both rungs
//!
//! On the dynamic-rendering rung there is no `VkRenderPass` at all: a pipeline
//! carries `VkPipelineRenderingCreateInfo`, which names the same formats and
//! the same sample count. So [`Compatibility`] is the pipeline cache key on
//! *both* rungs, and [`Compatibility::rendering_info`] is how it becomes the
//! one the dynamic rung wants. That is what makes [`crate::pass::select`] a
//! choice of rung rather than two pipelines caches with two key types.
//!
//! # One sample count for the whole pass
//!
//! Metal requires every attachment of a render pass to have the same sample
//! count, and Vulkan requires the same of a subpass — with the single
//! exception of resolve targets, which are always single-sampled. So
//! [`Compatibility`] holds one count rather than one per attachment, and a set
//! of attachments that disagree is refused by name rather than resolved to one
//! of them.
//!
//! # A framebuffer is not a pass
//!
//! `VkFramebuffer` binds *images* to a pass, so it is keyed on the views and
//! the extent as well as the pass — three attachments to the same textures at
//! different mip levels are three framebuffers over one render pass. Kept as
//! its own key for that reason, and because a framebuffer is invalidated by a
//! resource replacement while the pass it belongs to is not.
//!
//! # Built, not created
//!
//! Nothing here calls `vkCreateRenderPass` or `vkCmdBeginRendering`. Each
//! build owns the arrays its create info points at and hands out a borrowed
//! info, so the structure a driver would receive is inspectable with no GPU.

use crate::pass::{ClearColor, Ops, PassPlan};
use ash::vk;

/// What two passes must agree on for a pipeline built against one to run in
/// the other.
///
/// Deliberately holds no operation and no layout: those are not part of
/// Vulkan's compatibility rule, and including them would make this a second
/// spelling of [`Signature`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Compatibility {
    /// Colour attachment formats, in the pass's own attachment order.
    pub color: Vec<vk::Format>,
    /// `None` when the pass has no depth-stencil attachment, and otherwise the
    /// one format that carries both aspects the guest attached.
    pub depth_stencil: Option<vk::Format>,
    /// Whether the depth-stencil attachment's depth aspect is used, which
    /// `VkPipelineRenderingCreateInfo` spells as a format per aspect.
    pub depth: bool,
    pub stencil: bool,
    /// One count for the whole pass. See the module doc.
    pub samples: vk::SampleCountFlags,
}

impl Compatibility {
    /// The structure a pipeline is built with on the dynamic-rendering rung.
    ///
    /// The borrow is this value's: the create info points at
    /// [`Self::color`], so it may not outlive it.
    pub fn rendering_info(&self) -> vk::PipelineRenderingCreateInfo<'_> {
        let mut info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&self.color);
        // Per aspect, because an attachment carrying only stencil must leave
        // the depth format undefined — naming it would declare a depth
        // attachment the pass does not have.
        if let Some(format) = self.depth_stencil {
            if self.depth {
                info = info.depth_attachment_format(format);
            }
            if self.stencil {
                info = info.stencil_attachment_format(format);
            }
        }
        info
    }
}

/// One attachment's operations, as `VkAttachmentDescription` holds them.
///
/// Separate from [`Compatibility`] for the reason the module doc gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttachmentOps {
    pub load: vk::AttachmentLoadOp,
    pub store: vk::AttachmentStoreOp,
    pub stencil_load: vk::AttachmentLoadOp,
    pub stencil_store: vk::AttachmentStoreOp,
    pub initial_layout: vk::ImageLayout,
    pub final_layout: vk::ImageLayout,
}

/// Everything a `VkRenderPass` is created from.
///
/// Contains its [`Compatibility`] rather than restating the formats, so a
/// cache lookup on one and a pipeline lookup on the other cannot disagree.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Signature {
    pub compatibility: Compatibility,
    /// One per colour attachment, in the same order.
    pub color: Vec<AttachmentOps>,
    pub depth_stencil: Option<AttachmentOps>,
    /// Whether each colour attachment resolves. A resolve target is an extra
    /// attachment in the description, so this changes the object.
    pub resolve: Vec<bool>,
}

/// Everything a `VkFramebuffer` is created from.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FramebufferKey {
    pub render_pass: vk::RenderPass,
    /// In attachment order: every colour attachment, then its resolve target
    /// where it has one, then the depth-stencil attachment.
    pub views: Vec<vk::ImageView>,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
}

/// One attachment's image, resolved by the caller from the plan's resource
/// names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bound {
    pub format: vk::Format,
    pub samples: vk::SampleCountFlags,
    pub view: vk::ImageView,
    /// The resolve target's view. Required exactly when the plan's attachment
    /// asks to resolve.
    pub resolve_view: Option<vk::ImageView>,
}

/// Why a pass cannot be built here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The caller supplied a different number of images than the plan has
    /// colour attachments.
    ColorCountMismatch { planned: usize, bound: usize },
    /// The plan has a depth-stencil attachment and no image was supplied, or
    /// the reverse.
    DepthStencilMismatch { planned: bool, bound: bool },
    /// Attachments disagree about how many samples they have. See the module
    /// doc.
    SampleCountMismatch {
        first: vk::SampleCountFlags,
        found: vk::SampleCountFlags,
    },
    /// The plan asks this attachment to resolve and no resolve image was
    /// supplied — or one was supplied for an attachment that does not resolve.
    ResolveMismatch { index: usize, planned: bool },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::ColorCountMismatch { .. } => "vk_renderpass_color_count",
            Self::DepthStencilMismatch { .. } => "vk_renderpass_depth_stencil_count",
            Self::SampleCountMismatch { .. } => "vk_renderpass_sample_count",
            Self::ResolveMismatch { .. } => "vk_renderpass_resolve",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ColorCountMismatch { planned, bound } => {
                write!(f, "{} planned={planned} bound={bound}", self.slug())
            }
            Self::DepthStencilMismatch { planned, bound } => {
                write!(f, "{} planned={planned} bound={bound}", self.slug())
            }
            Self::SampleCountMismatch { first, found } => write!(
                f,
                "{} first={:?} found={:?}",
                self.slug(),
                first.as_raw(),
                found.as_raw()
            ),
            Self::ResolveMismatch { index, planned } => {
                write!(f, "{} index={index} planned={planned}", self.slug())
            }
        }
    }
}

/// The layout an attachment is in for the whole pass.
///
/// One layout for the initial and the final, because this rail transitions
/// attachments with explicit barriers around the pass rather than letting the
/// pass do it: a pass that changed the layout would move an image the layout
/// tracker believes is somewhere else, and the tracker is what every other
/// barrier is planned against. `VK_IMAGE_LAYOUT_UNDEFINED` as an initial
/// layout would additionally discard the contents a `LOAD` attachment is
/// loading.
const fn color_layout() -> vk::ImageLayout {
    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
}

const fn depth_stencil_layout() -> vk::ImageLayout {
    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
}

/// A pass, built into everything either rung needs.
///
/// Owns the arrays the create infos point at, so a borrowed info is valid for
/// as long as this value is.
///
/// `Debug` is written rather than derived: `VkClearValue` is a union and
/// `VkRenderingAttachmentInfo` carries a raw `p_next`, and neither has one.
/// What a report wants is the signature and the shape, which is what it
/// prints.
#[derive(Clone)]
pub struct Build {
    signature: Signature,
    attachments: Vec<vk::AttachmentDescription>,
    color_refs: Vec<vk::AttachmentReference>,
    resolve_refs: Vec<vk::AttachmentReference>,
    depth_ref: Option<vk::AttachmentReference>,
    views: Vec<vk::ImageView>,
    clears: Vec<vk::ClearValue>,
    color_rendering: Vec<vk::RenderingAttachmentInfo<'static>>,
    depth_rendering: Option<vk::RenderingAttachmentInfo<'static>>,
    stencil_rendering: Option<vk::RenderingAttachmentInfo<'static>>,
    area: vk::Rect2D,
    layers: u32,
}

impl std::fmt::Debug for Build {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Build")
            .field("signature", &self.signature)
            .field("attachments", &self.attachments.len())
            .field("views", &self.views)
            .field("area", &self.area)
            .field("layers", &self.layers)
            .finish()
    }
}

fn ops_of(
    load: vk::AttachmentLoadOp,
    store: vk::AttachmentStoreOp,
    layout: vk::ImageLayout,
) -> AttachmentOps {
    AttachmentOps {
        load,
        store,
        // A colour attachment has no stencil aspect; Vulkan ignores these and
        // `DONT_CARE` is the value that says so rather than one that claims a
        // load nothing performs.
        stencil_load: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: layout,
        final_layout: layout,
    }
}

fn attachment_description(
    format: vk::Format,
    samples: vk::SampleCountFlags,
    ops: AttachmentOps,
) -> vk::AttachmentDescription {
    vk::AttachmentDescription {
        flags: vk::AttachmentDescriptionFlags::empty(),
        format,
        samples,
        load_op: ops.load,
        store_op: ops.store,
        stencil_load_op: ops.stencil_load,
        stencil_store_op: ops.stencil_store,
        initial_layout: ops.initial_layout,
        final_layout: ops.final_layout,
    }
}

fn rendering_attachment(
    view: vk::ImageView,
    layout: vk::ImageLayout,
    load: vk::AttachmentLoadOp,
    store: vk::AttachmentStoreOp,
    clear: vk::ClearValue,
    resolve: Option<vk::ImageView>,
) -> vk::RenderingAttachmentInfo<'static> {
    let mut info = vk::RenderingAttachmentInfo::default()
        .image_view(view)
        .image_layout(layout)
        .load_op(load)
        .store_op(store)
        .clear_value(clear);
    if let Some(target) = resolve {
        info = info
            .resolve_mode(vk::ResolveModeFlags::AVERAGE)
            .resolve_image_view(target)
            .resolve_image_layout(layout);
    }
    info
}

/// The clear a plan carries for one colour attachment, or a zero the driver
/// ignores because the load operation is not a clear.
fn color_clear(clear: Option<ClearColor>) -> vk::ClearValue {
    vk::ClearValue {
        color: clear.map_or(
            vk::ClearColorValue { float32: [0.0; 4] },
            ClearColor::native,
        ),
    }
}

/// Build a planned pass against the images the caller resolved for it.
///
/// A [`PassPlan`] with no attachment at all cannot arrive here: [`crate::pass::plan`]
/// refuses that descriptor, and re-checking it would put one invariant in two
/// places with only one of them tested against a real descriptor.
///
/// # Errors
///
/// [`Refusal`], with nothing partially built.
pub fn build(
    plan: &PassPlan,
    color: &[Bound],
    depth_stencil: Option<Bound>,
) -> Result<Build, Refusal> {
    if plan.color.len() != color.len() {
        return Err(Refusal::ColorCountMismatch {
            planned: plan.color.len(),
            bound: color.len(),
        });
    }
    if plan.depth_stencil.is_some() != depth_stencil.is_some() {
        return Err(Refusal::DepthStencilMismatch {
            planned: plan.depth_stencil.is_some(),
            bound: depth_stencil.is_some(),
        });
    }

    // One count for the whole pass, taken from the first attachment and
    // checked against every other. Resolve targets are excluded because they
    // are single-sampled by definition.
    let mut samples: Option<vk::SampleCountFlags> = None;
    for bound in color.iter().chain(depth_stencil.iter()) {
        match samples {
            None => samples = Some(bound.samples),
            Some(first) if first != bound.samples => {
                return Err(Refusal::SampleCountMismatch {
                    first,
                    found: bound.samples,
                })
            }
            Some(_) => {}
        }
    }
    let samples = samples.unwrap_or(vk::SampleCountFlags::TYPE_1);

    for (index, (planned, bound)) in plan.color.iter().zip(color).enumerate() {
        if planned.resolve.is_some() != bound.resolve_view.is_some() {
            return Err(Refusal::ResolveMismatch {
                index,
                planned: planned.resolve.is_some(),
            });
        }
    }

    let mut attachments = Vec::new();
    let mut color_refs = Vec::new();
    let mut resolve_refs = Vec::new();
    let mut views = Vec::new();
    let mut clears = Vec::new();
    let mut color_ops = Vec::new();
    let mut resolve_flags = Vec::new();
    let mut color_rendering = Vec::new();

    for (planned, bound) in plan.color.iter().zip(color) {
        let ops = ops_of(planned.load, planned.store, color_layout());
        color_refs.push(vk::AttachmentReference {
            attachment: u32::try_from(attachments.len()).unwrap_or(u32::MAX),
            layout: color_layout(),
        });
        attachments.push(attachment_description(bound.format, samples, ops));
        views.push(bound.view);
        let clear = color_clear(planned.clear);
        clears.push(clear);
        color_ops.push(ops);
        resolve_flags.push(planned.resolve.is_some());
        color_rendering.push(rendering_attachment(
            bound.view,
            color_layout(),
            planned.load,
            planned.store,
            clear,
            bound.resolve_view,
        ));

        if let Some(target) = bound.resolve_view {
            resolve_refs.push(vk::AttachmentReference {
                attachment: u32::try_from(attachments.len()).unwrap_or(u32::MAX),
                layout: color_layout(),
            });
            // A resolve target is written and not read, and it is always
            // single-sampled whatever the pass is.
            attachments.push(attachment_description(
                bound.format,
                vk::SampleCountFlags::TYPE_1,
                ops_of(
                    vk::AttachmentLoadOp::DONT_CARE,
                    vk::AttachmentStoreOp::STORE,
                    color_layout(),
                ),
            ));
            views.push(target);
            clears.push(vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0; 4] },
            });
        } else {
            // Vulkan wants either no resolve array or one entry per colour
            // attachment, so an attachment that does not resolve contributes
            // `VK_ATTACHMENT_UNUSED` rather than being skipped — skipping it
            // would shift every later attachment's resolve target onto the
            // wrong one.
            resolve_refs.push(vk::AttachmentReference {
                attachment: vk::ATTACHMENT_UNUSED,
                layout: vk::ImageLayout::UNDEFINED,
            });
        }
    }

    let mut depth_ref = None;
    let mut depth_stencil_ops = None;
    let mut depth_rendering = None;
    let mut stencil_rendering = None;
    let mut depth_stencil_format = None;
    let (mut has_depth, mut has_stencil) = (false, false);

    if let (Some(planned), Some(bound)) = (plan.depth_stencil.as_ref(), depth_stencil) {
        let depth = planned.depth.unwrap_or(Ops {
            load: vk::AttachmentLoadOp::DONT_CARE,
            store: vk::AttachmentStoreOp::DONT_CARE,
        });
        let stencil = planned.stencil.unwrap_or(Ops {
            load: vk::AttachmentLoadOp::DONT_CARE,
            store: vk::AttachmentStoreOp::DONT_CARE,
        });
        let ops = AttachmentOps {
            load: depth.load,
            store: depth.store,
            stencil_load: stencil.load,
            stencil_store: stencil.store,
            initial_layout: depth_stencil_layout(),
            final_layout: depth_stencil_layout(),
        };
        depth_ref = Some(vk::AttachmentReference {
            attachment: u32::try_from(attachments.len()).unwrap_or(u32::MAX),
            layout: depth_stencil_layout(),
        });
        attachments.push(attachment_description(bound.format, samples, ops));
        views.push(bound.view);
        let clear = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: planned.clear_depth,
                stencil: planned.clear_stencil,
            },
        };
        clears.push(clear);
        depth_stencil_ops = Some(ops);
        depth_stencil_format = Some(bound.format);
        has_depth = planned.depth.is_some();
        has_stencil = planned.stencil.is_some();
        if has_depth {
            depth_rendering = Some(rendering_attachment(
                bound.view,
                depth_stencil_layout(),
                depth.load,
                depth.store,
                clear,
                None,
            ));
        }
        if has_stencil {
            stencil_rendering = Some(rendering_attachment(
                bound.view,
                depth_stencil_layout(),
                stencil.load,
                stencil.store,
                clear,
                None,
            ));
        }
    }

    let signature = Signature {
        compatibility: Compatibility {
            color: color.iter().map(|b| b.format).collect(),
            depth_stencil: depth_stencil_format,
            depth: has_depth,
            stencil: has_stencil,
            samples,
        },
        color: color_ops,
        depth_stencil: depth_stencil_ops,
        resolve: resolve_flags,
    };

    Ok(Build {
        signature,
        attachments,
        color_refs,
        resolve_refs,
        depth_ref,
        views,
        clears,
        color_rendering,
        depth_rendering,
        stencil_rendering,
        area: vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: plan.extent,
        },
        layers: plan.layers,
    })
}

impl Build {
    /// The key a `VkRenderPass` cache stores this under.
    #[must_use]
    pub fn signature(&self) -> &Signature {
        &self.signature
    }

    /// The key a *pipeline* cache stores against this pass, on either rung.
    #[must_use]
    pub fn compatibility(&self) -> &Compatibility {
        &self.signature.compatibility
    }

    /// The key a `VkFramebuffer` cache stores this under, given the pass it
    /// was created against.
    #[must_use]
    pub fn framebuffer_key(&self, render_pass: vk::RenderPass) -> FramebufferKey {
        FramebufferKey {
            render_pass,
            views: self.views.clone(),
            width: self.area.extent.width,
            height: self.area.extent.height,
            layers: self.layers,
        }
    }

    /// Every attachment description, in the order the references index.
    pub fn attachments(&self) -> &[vk::AttachmentDescription] {
        &self.attachments
    }

    /// The clear values a `vkCmdBeginRenderPass` takes, one per attachment and
    /// in the same order.
    #[must_use]
    pub fn clear_values(&self) -> &[vk::ClearValue] {
        &self.clears
    }

    /// Whether any colour attachment resolves, which is what decides whether
    /// the subpass carries a resolve array at all.
    #[must_use]
    pub fn resolves(&self) -> bool {
        self.signature.resolve.iter().any(|r| *r)
    }

    /// Hand `f` the `VkRenderPassCreateInfo` this build describes.
    ///
    /// A closure rather than a returned structure because the subpass
    /// description points at arrays this value owns, and a returned info would
    /// have to carry a lifetime through the subpass as well.
    pub fn with_render_pass_create_info<R>(
        &self,
        f: impl FnOnce(&vk::RenderPassCreateInfo<'_>) -> R,
    ) -> R {
        let mut subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&self.color_refs);
        if self.resolves() {
            subpass = subpass.resolve_attachments(&self.resolve_refs);
        }
        if let Some(reference) = self.depth_ref.as_ref() {
            subpass = subpass.depth_stencil_attachment(reference);
        }
        let subpasses = [subpass];
        let info = vk::RenderPassCreateInfo::default()
            .attachments(&self.attachments)
            .subpasses(&subpasses);
        f(&info)
    }

    /// Hand `f` the `VkFramebufferCreateInfo` for `render_pass`.
    pub fn with_framebuffer_create_info<R>(
        &self,
        render_pass: vk::RenderPass,
        f: impl FnOnce(&vk::FramebufferCreateInfo<'_>) -> R,
    ) -> R {
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&self.views)
            .width(self.area.extent.width)
            .height(self.area.extent.height)
            .layers(self.layers);
        f(&info)
    }

    /// Hand `f` the `VkRenderingInfo` for `vkCmdBeginRendering`.
    pub fn with_rendering_info<R>(&self, f: impl FnOnce(&vk::RenderingInfo<'_>) -> R) -> R {
        let mut info = vk::RenderingInfo::default()
            .render_area(self.area)
            .layer_count(self.layers)
            .color_attachments(&self.color_rendering);
        if let Some(depth) = self.depth_rendering.as_ref() {
            info = info.depth_attachment(depth);
        }
        if let Some(stencil) = self.stencil_rendering.as_ref() {
            info = info.stencil_attachment(stencil);
        }
        f(&info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass::plan;
    use ash::vk::Handle;
    use reims_vgpu_core::identity::{ObjectListRef, ResourceId, SlotGeneration};
    use reims_vgpu_core::pass::{LoadAction, StoreAction};
    use reims_vgpu_core::pass::{PassDescriptor, RenderTargetExtent};
    use reims_vgpu_core::pixel_format::{MTL_FORMAT_DEPTH32_FLOAT, MTL_FORMAT_RGBA8_UNORM};
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn id(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn view(n: u64) -> vk::ImageView {
        vk::ImageView::from_raw(n)
    }

    fn descriptor() -> PassDescriptor {
        let mut d = PassDescriptor::empty();
        d.extent = RenderTargetExtent {
            width: 128,
            height: 64,
            array_length: 1,
        };
        d
    }

    fn one_colour(load: LoadAction, store: StoreAction) -> PassPlan {
        let mut d = descriptor();
        d.color[0].texture = Some(id(1));
        d.color[0].load = load;
        d.color[0].store = store;
        plan(&d, |_| MTL_FORMAT_RGBA8_UNORM).expect("a legal descriptor")
    }

    fn bound(view_id: u64) -> Bound {
        Bound {
            format: vk::Format::R8G8B8A8_UNORM,
            samples: vk::SampleCountFlags::TYPE_1,
            view: view(view_id),
            resolve_view: None,
        }
    }

    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    /// The claim the module exists for: two passes that differ only in a load
    /// action are two render passes and one pipeline.
    #[test]
    fn a_load_action_changes_the_pass_object_and_not_the_pipeline_key() {
        let clearing = build(
            &one_colour(LoadAction::Clear, StoreAction::Store),
            &[bound(1)],
            None,
        )
        .expect("legal");
        let loading = build(
            &one_colour(LoadAction::Load, StoreAction::Store),
            &[bound(1)],
            None,
        )
        .expect("legal");

        assert_ne!(clearing.signature(), loading.signature());
        assert_eq!(clearing.compatibility(), loading.compatibility());
        // And the keys hash apart and together the same way, because a cache
        // looks them up by hash before it compares them.
        assert_ne!(hash_of(clearing.signature()), hash_of(loading.signature()));
        assert_eq!(
            hash_of(clearing.compatibility()),
            hash_of(loading.compatibility())
        );

        // The difference really is the load op and nothing else.
        assert_eq!(
            clearing.attachments()[0].load_op,
            vk::AttachmentLoadOp::CLEAR
        );
        assert_eq!(loading.attachments()[0].load_op, vk::AttachmentLoadOp::LOAD);
        assert_eq!(
            clearing.attachments()[0].format,
            loading.attachments()[0].format
        );
    }

    /// A store action is not part of compatibility either, and a *format*
    /// change is.
    #[test]
    fn the_pipeline_key_moves_with_a_format_and_not_with_a_store() {
        let stored = build(
            &one_colour(LoadAction::Load, StoreAction::Store),
            &[bound(1)],
            None,
        )
        .expect("legal");
        let dropped = build(
            &one_colour(LoadAction::Load, StoreAction::DontCare),
            &[bound(1)],
            None,
        )
        .expect("legal");
        assert_eq!(stored.compatibility(), dropped.compatibility());
        assert_ne!(stored.signature(), dropped.signature());

        let other_format = build(
            &one_colour(LoadAction::Load, StoreAction::Store),
            &[Bound {
                format: vk::Format::R16G16B16A16_SFLOAT,
                ..bound(1)
            }],
            None,
        )
        .expect("legal");
        assert_ne!(stored.compatibility(), other_format.compatibility());

        // As is a sample count.
        let multisampled = build(
            &one_colour(LoadAction::Load, StoreAction::Store),
            &[Bound {
                samples: vk::SampleCountFlags::TYPE_4,
                ..bound(1)
            }],
            None,
        )
        .expect("legal");
        assert_ne!(stored.compatibility(), multisampled.compatibility());
    }

    /// The framebuffer key moves with the images while the pass does not:
    /// three attachments to three textures are one render pass.
    #[test]
    fn the_framebuffer_key_moves_with_the_views_and_the_pass_does_not() {
        let plan = one_colour(LoadAction::Load, StoreAction::Store);
        let first = build(&plan, &[bound(1)], None).expect("legal");
        let second = build(&plan, &[bound(2)], None).expect("legal");
        assert_eq!(first.signature(), second.signature());

        let pass = vk::RenderPass::from_raw(7);
        assert_ne!(first.framebuffer_key(pass), second.framebuffer_key(pass));
        assert_eq!(first.framebuffer_key(pass), first.framebuffer_key(pass));
        // And the pass is part of the key, because a framebuffer belongs to
        // one.
        assert_ne!(
            first.framebuffer_key(pass),
            first.framebuffer_key(vk::RenderPass::from_raw(8))
        );
        let key = first.framebuffer_key(pass);
        assert_eq!(key.views, vec![view(1)]);
        assert_eq!((key.width, key.height, key.layers), (128, 64, 1));
    }

    #[test]
    fn the_render_pass_create_info_names_every_attachment_once() {
        let mut d = descriptor();
        d.color[0].texture = Some(id(1));
        d.color[0].load = LoadAction::Clear;
        d.color[0].store = StoreAction::Store;
        d.color[2].texture = Some(id(2));
        d.color[2].load = LoadAction::Load;
        d.color[2].store = StoreAction::Store;
        d.depth.texture = Some(id(3));
        d.depth.load = LoadAction::Clear;
        d.depth.store = StoreAction::Store;
        let plan = plan(&d, |r| {
            if r == id(3) {
                MTL_FORMAT_DEPTH32_FLOAT
            } else {
                MTL_FORMAT_RGBA8_UNORM
            }
        })
        .expect("a legal descriptor");

        let depth = Bound {
            format: vk::Format::D32_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            view: view(30),
            resolve_view: None,
        };
        let built = build(&plan, &[bound(10), bound(20)], Some(depth)).expect("legal");

        // The guest's slot two is Vulkan's colour attachment one: a gap in the
        // guest's slots renumbers, and it renumbers exactly once.
        assert_eq!(built.compatibility().color.len(), 2);
        assert_eq!(built.attachments().len(), 3);
        assert_eq!(built.clear_values().len(), 3);
        assert!(!built.resolves());

        built.with_render_pass_create_info(|info| {
            assert_eq!(info.attachment_count, 3);
            assert_eq!(info.subpass_count, 1);
            // SAFETY: the pointer is into arrays `built` owns and outlives.
            let subpass = unsafe { &*info.p_subpasses };
            assert_eq!(subpass.color_attachment_count, 2);
            assert!(!subpass.p_depth_stencil_attachment.is_null());
            // No resolve array where nothing resolves.
            assert!(subpass.p_resolve_attachments.is_null());
            // SAFETY: two colour references were pushed.
            let refs = unsafe { std::slice::from_raw_parts(subpass.p_color_attachments, 2) };
            assert_eq!(refs[0].attachment, 0);
            assert_eq!(refs[1].attachment, 1);
            // SAFETY: the depth reference is non-null, checked above.
            let depth_ref = unsafe { &*subpass.p_depth_stencil_attachment };
            assert_eq!(depth_ref.attachment, 2);
        });

        built.with_framebuffer_create_info(vk::RenderPass::from_raw(1), |info| {
            assert_eq!(info.attachment_count, 3);
            assert_eq!((info.width, info.height, info.layers), (128, 64, 1));
        });
    }

    /// An attachment that does not resolve still occupies a slot in the
    /// resolve array, or every later attachment's target shifts onto the wrong
    /// one.
    #[test]
    fn a_resolve_array_is_one_entry_per_colour_attachment_or_none_at_all() {
        let mut d = descriptor();
        for slot in 0..2 {
            d.color[slot].texture = Some(id(slot as u32 + 1));
            d.color[slot].load = LoadAction::Load;
            d.color[slot].store = StoreAction::Store;
        }
        d.color[1].store = StoreAction::MultisampleResolve;
        d.color[1].resolve_texture = Some(id(9));
        let plan = plan(&d, |_| MTL_FORMAT_RGBA8_UNORM).expect("a legal descriptor");

        let resolving = Bound {
            resolve_view: Some(view(99)),
            samples: vk::SampleCountFlags::TYPE_4,
            ..bound(2)
        };
        let plain = Bound {
            samples: vk::SampleCountFlags::TYPE_4,
            ..bound(1)
        };
        let built = build(&plan, &[plain, resolving], None).expect("legal");
        assert!(built.resolves());
        // Two colour attachments plus one resolve target.
        assert_eq!(built.attachments().len(), 3);
        // The resolve target is single-sampled whatever the pass is.
        assert_eq!(built.compatibility().samples, vk::SampleCountFlags::TYPE_4);
        assert_eq!(built.attachments()[2].samples, vk::SampleCountFlags::TYPE_1);

        built.with_render_pass_create_info(|info| {
            // SAFETY: one subpass was pushed.
            let subpass = unsafe { &*info.p_subpasses };
            assert!(!subpass.p_resolve_attachments.is_null());
            // SAFETY: the array has one entry per colour attachment.
            let resolves = unsafe { std::slice::from_raw_parts(subpass.p_resolve_attachments, 2) };
            assert_eq!(resolves[0].attachment, vk::ATTACHMENT_UNUSED);
            assert_eq!(resolves[1].attachment, 2);
        });
    }

    /// The dynamic rung takes the same compatibility key and turns it into the
    /// structure its pipelines are built with.
    #[test]
    fn the_dynamic_rung_builds_a_pipeline_from_the_same_key() {
        let mut d = descriptor();
        d.color[0].texture = Some(id(1));
        d.color[0].load = LoadAction::Clear;
        d.color[0].store = StoreAction::Store;
        d.depth.texture = Some(id(2));
        d.depth.load = LoadAction::Clear;
        d.depth.store = StoreAction::Store;
        let plan = plan(&d, |r| {
            if r == id(2) {
                MTL_FORMAT_DEPTH32_FLOAT
            } else {
                MTL_FORMAT_RGBA8_UNORM
            }
        })
        .expect("a legal descriptor");
        let depth = Bound {
            format: vk::Format::D32_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            view: view(2),
            resolve_view: None,
        };
        let built = build(&plan, &[bound(1)], Some(depth)).expect("legal");

        let compatibility = built.compatibility();
        assert!(compatibility.depth);
        // A depth-only attachment leaves the stencil format undefined, so the
        // pipeline does not declare a stencil attachment the pass lacks.
        assert!(!compatibility.stencil);
        let info = compatibility.rendering_info();
        assert_eq!(info.color_attachment_count, 1);
        assert_eq!(info.depth_attachment_format, vk::Format::D32_SFLOAT);
        assert_eq!(info.stencil_attachment_format, vk::Format::UNDEFINED);

        built.with_rendering_info(|info| {
            assert_eq!(info.color_attachment_count, 1);
            assert_eq!(info.layer_count, 1);
            assert_eq!(
                info.render_area.extent,
                vk::Extent2D {
                    width: 128,
                    height: 64,
                }
            );
            assert!(!info.p_depth_attachment.is_null());
            assert!(info.p_stencil_attachment.is_null());
            // SAFETY: the colour array has one entry.
            let color = unsafe { std::slice::from_raw_parts(info.p_color_attachments, 1) };
            assert_eq!(color[0].image_view, view(1));
            assert_eq!(color[0].load_op, vk::AttachmentLoadOp::CLEAR);
            assert_eq!(
                color[0].image_layout,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            );
        });
    }

    /// Attachments that disagree about sample count are refused, not resolved
    /// to one of them.
    #[test]
    fn attachments_that_disagree_about_samples_refuse() {
        let mut d = descriptor();
        for slot in 0..2 {
            d.color[slot].texture = Some(id(slot as u32 + 1));
            d.color[slot].load = LoadAction::Load;
            d.color[slot].store = StoreAction::Store;
        }
        let plan = plan(&d, |_| MTL_FORMAT_RGBA8_UNORM).expect("a legal descriptor");
        let refused = build(
            &plan,
            &[
                bound(1),
                Bound {
                    samples: vk::SampleCountFlags::TYPE_4,
                    ..bound(2)
                },
            ],
            None,
        )
        .expect_err("one pass, one sample count");
        assert_eq!(
            refused,
            Refusal::SampleCountMismatch {
                first: vk::SampleCountFlags::TYPE_1,
                found: vk::SampleCountFlags::TYPE_4,
            }
        );
        assert_eq!(refused.slug(), "vk_renderpass_sample_count");

        // Agreeing at four is fine, so the refusal is about disagreement and
        // not about multisampling.
        assert!(build(
            &plan,
            &[
                Bound {
                    samples: vk::SampleCountFlags::TYPE_4,
                    ..bound(1)
                },
                Bound {
                    samples: vk::SampleCountFlags::TYPE_4,
                    ..bound(2)
                },
            ],
            None,
        )
        .is_ok());
    }

    #[test]
    fn a_caller_that_supplied_the_wrong_images_is_refused_by_name() {
        let plan = one_colour(LoadAction::Load, StoreAction::Store);
        assert_eq!(
            build(&plan, &[], None).expect_err("no images for one attachment"),
            Refusal::ColorCountMismatch {
                planned: 1,
                bound: 0
            }
        );
        assert_eq!(
            build(&plan, &[bound(1)], Some(bound(2)))
                .expect_err("a depth image for a pass with no depth"),
            Refusal::DepthStencilMismatch {
                planned: false,
                bound: true
            }
        );
        // A colour attachment that does not resolve, handed a resolve image.
        assert_eq!(
            build(
                &plan,
                &[Bound {
                    resolve_view: Some(view(9)),
                    ..bound(1)
                }],
                None
            )
            .expect_err("a resolve image for an attachment that does not resolve"),
            Refusal::ResolveMismatch {
                index: 0,
                planned: false
            }
        );

        // A pass with nothing attached never reaches here: the plan owns that
        // refusal, and this is the assertion that keeps it owned there.
        assert!(crate::pass::plan(&descriptor(), |_| MTL_FORMAT_RGBA8_UNORM).is_err());

        let slugs: std::collections::BTreeSet<&str> = [
            Refusal::ColorCountMismatch {
                planned: 1,
                bound: 0,
            },
            Refusal::DepthStencilMismatch {
                planned: true,
                bound: false,
            },
            Refusal::SampleCountMismatch {
                first: vk::SampleCountFlags::TYPE_1,
                found: vk::SampleCountFlags::TYPE_2,
            },
            Refusal::ResolveMismatch {
                index: 0,
                planned: true,
            },
        ]
        .iter()
        .map(|r| r.slug())
        .collect();
        assert_eq!(slugs.len(), 4);
    }
}
