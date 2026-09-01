//! A graphics pipeline: the one object every fixed-function plan on this rail
//! is finally assembled into.
//!
//! Everything here is already translated. [`crate::vertex`],
//! [`crate::topology`], [`crate::raster`], [`crate::depth_stencil`],
//! [`crate::blend`] and [`crate::renderpass`] each own one part of the state
//! and each
//! refuses what it cannot serve; this module owns only their *composition*,
//! and the two facts that composition creates.
//!
//! # A key is what a built pipeline can serve, not what a draw asked for
//!
//! [`GraphicsKey`] is the cache dimension, and every part of it has already
//! had this host's dynamic state spent on it: [`topology::TopologyKey`]
//! collapses primitive types the host can change per draw,
//! [`raster::RasterizationState`] holds a default wherever the encoder
//! supplies the guest's value, and [`blend`] fixes a disabled attachment's
//! factors. So two draws whose keys are equal are two draws one compilation
//! serves — that is the claim the key makes, and it is why the key is built
//! from the plans rather than from the guest state they came from.
//!
//! Compilation is the most expensive thing this rail does, so the key must not
//! carry anything a draw can vary without a rebuild. It carries no viewport,
//! no scissor rectangle, no blend colour, no stencil reference and no depth
//! bias — all six are dynamic on every device Vulkan admits, because all six
//! are encoder commands in Metal and a pipeline per value would be a pipeline
//! per frame.
//!
//! # The render pass is a compatibility class, on both carriers
//!
//! A pipeline is built against a pass, and Vulkan's rule is *compatibility*
//! rather than identity: two render passes with the same attachment formats
//! and sample count serve each other's pipelines whatever their load and store
//! operations are. [`renderpass::Compatibility`] is exactly that class, which
//! is why it and not [`renderpass::Signature`] is the key here — keying on the
//! signature would compile a second pipeline for a pass that differs only in
//! clearing where the first loaded.
//!
//! The same class is the key on the dynamic-rendering carrier, where there is
//! no render-pass object at all and `VkPipelineRenderingCreateInfo` names the
//! formats directly. Deliberately the same type on both rungs: the two
//! carriers must not disagree about which draws share a pipeline, or a host
//! that took the other rung would compile a different number of them for one
//! guest.
//!
//! # What this module refuses
//!
//! Only composition failures — a state that is individually translatable and
//! wrong beside another. A colour blend attachment for a pass with no such
//! attachment; a depth-stencil state whose pass has no depth-stencil
//! attachment, or a pass that has one with no state to drive it; a pipeline
//! with no vertex stage. Each of those produces either a validation error or,
//! worse, a silently ignored piece of guest state.

use ash::vk;
use std::ffi::CStr;

use crate::{blend, depth_stencil, raster, renderpass, topology, vertex};

/// One shader stage, as far as identity is concerned.
///
/// The module handle and the entry point, and nothing else: two pipelines
/// built from the same module and the same entry run the same code, and a
/// caller that recreates a module gets a new handle and therefore a new key,
/// which is the answer that keeps a stale pipeline from serving new code.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StageKey {
    pub stage: vk::ShaderStageFlags,
    pub module: vk::ShaderModule,
    /// The entry point, without its terminator.
    pub entry: String,
}

/// The multisample state, which no other module owns.
///
/// The sample count is not here: it is [`renderpass::Compatibility`]'s, one
/// count for the whole pass, and a second copy could disagree with the pass
/// the pipeline is built against.
///
/// Both flags are `MTLRenderPipelineDescriptor` fields with no ordinal to
/// parse and no capability to check — Vulkan 1.0 carries `alphaToCoverage` and
/// `alphaToOne` in the same structure — so they are taken as declared.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MultisamplePlan {
    pub alpha_to_coverage: bool,
    pub alpha_to_one: bool,
}

/// What identifies a graphics pipeline on this host.
///
/// Two draws with equal keys share one compilation. See the module doc for
/// what is deliberately absent.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GraphicsKey {
    /// In the order they are declared to Vulkan.
    pub stages: Vec<StageKey>,
    pub layout: vk::PipelineLayout,
    pub bindings: Vec<vertex::BindingPlan>,
    pub attributes: Vec<vertex::AttributePlan>,
    pub topology: topology::TopologyKey,
    pub raster: raster::RasterizationState,
    pub multisample: MultisamplePlan,
    /// `None` for a pass with no depth-stencil attachment. Must agree with
    /// [`renderpass::Compatibility::depth_stencil`]; [`build`] refuses a
    /// disagreement.
    pub depth_stencil: Option<depth_stencil::DepthStencilPlan>,
    /// One per colour attachment, in the pass's attachment order.
    pub blend: Vec<blend::AttachmentPlan>,
    /// The class of passes this pipeline runs in, on either carrier.
    pub compatibility: renderpass::Compatibility,
    /// How many viewports and scissor rectangles the pipeline declares.
    ///
    /// Their *values* are dynamic on every device, but the count is a pipeline
    /// member below `VK_EXT_extended_dynamic_state`'s with-count spellings, so
    /// a guest that binds three viewports and a guest that binds one are two
    /// pipelines here. One is overwhelmingly the common case, so this is a
    /// dimension that almost never divides.
    pub viewports: u32,
}

/// Why a set of individually translatable states cannot be one pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Nothing to run. A pipeline with no vertex stage has no way to produce
    /// the primitives its other stages consume.
    NoVertexStage,
    /// One stage declared twice. Vulkan forbids it, and the second would
    /// silently replace the first.
    DuplicateStage { stage: vk::ShaderStageFlags },
    /// A blend attachment count that is not the pass's colour count. Fewer
    /// leaves an attachment with no write mask; more names an attachment the
    /// pass does not have.
    BlendAttachmentCount { blend: usize, color: usize },
    /// A depth-stencil state for a pass with no depth-stencil attachment. The
    /// state would be ignored, so the guest's depth test would silently not
    /// run.
    DepthStateWithoutAttachment,
    /// A pass with a depth-stencil attachment and no state to drive it.
    /// Vulkan requires the structure, and its absence is undefined rather than
    /// a disabled test.
    AttachmentWithoutDepthState,
    /// No viewport at all. A pipeline must declare at least one, and a draw
    /// with none rasterizes nothing.
    NoViewport,
    /// A pass with no colour attachment and no depth-stencil attachment.
    /// Nothing to write to.
    NoAttachment,
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NoVertexStage => "vk_pipeline_no_vertex_stage",
            Self::DuplicateStage { .. } => "vk_pipeline_duplicate_stage",
            Self::BlendAttachmentCount { .. } => "vk_pipeline_blend_attachment_count",
            Self::DepthStateWithoutAttachment => "vk_pipeline_depth_state_without_attachment",
            Self::AttachmentWithoutDepthState => "vk_pipeline_attachment_without_depth_state",
            Self::NoViewport => "vk_pipeline_no_viewport",
            Self::NoAttachment => "vk_pipeline_no_attachment",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::DuplicateStage { stage } => {
                write!(f, "{} stage={:#x}", self.slug(), stage.as_raw())
            }
            Self::BlendAttachmentCount { blend, color } => {
                write!(f, "{} blend={blend} color={color}", self.slug())
            }
            _ => f.write_str(self.slug()),
        }
    }
}

/// Everything a `VkGraphicsPipeline` is created from, with the arrays its
/// create info points at.
///
/// Holds its [`GraphicsKey`] rather than restating any of it, so a cache
/// lookup and the object the lookup misses cannot disagree.
pub struct Build {
    key: GraphicsKey,
    entries: Vec<std::ffi::CString>,
    bindings: Vec<vk::VertexInputBindingDescription>,
    divisors: Vec<vk::VertexInputBindingDivisorDescriptionEXT>,
    attributes: Vec<vk::VertexInputAttributeDescription>,
    attachments: Vec<vk::PipelineColorBlendAttachmentState>,
    dynamic: Vec<vk::DynamicState>,
}

impl std::fmt::Debug for Build {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Build")
            .field("key", &self.key)
            .field("divisors", &self.divisors.len())
            .field("dynamic", &self.dynamic)
            .finish()
    }
}

/// The dynamic states every pipeline this rail builds declares, whatever the
/// host offers.
///
/// All four are Vulkan 1.0 dynamic states, so there is no capability to check
/// and no rung to fall off. All four are also encoder commands in Metal — a
/// guest may set any of them between two draws of one pass — so a pipeline
/// that baked one would be a pipeline per value. [`raster::RasterDynamic`]
/// adds `DEPTH_BIAS` to this list for the same reason, and the states it adds
/// beyond that are the ones that *do* need a capability.
const ALWAYS_DYNAMIC: [vk::DynamicState; 5] = [
    vk::DynamicState::VIEWPORT,
    vk::DynamicState::SCISSOR,
    vk::DynamicState::BLEND_CONSTANTS,
    vk::DynamicState::STENCIL_REFERENCE,
    // Metal's `setStencilFrontReferenceValue:` moves the reference; the two
    // masks are on the depth-stencil state object and are baked with it. So
    // this list has the reference and not the masks, which is the split
    // `depth_stencil::FacePlan` already makes by zeroing the reference.
    vk::DynamicState::DEPTH_BOUNDS,
];

/// Assemble a graphics pipeline from plans that have each already been checked.
///
/// # Errors
///
/// [`Refusal`] for a composition that no individual plan could have caught.
/// Nothing is partially built.
pub fn build(key: GraphicsKey) -> Result<Build, Refusal> {
    if key.viewports == 0 {
        return Err(Refusal::NoViewport);
    }
    if key.compatibility.color.is_empty() && key.compatibility.depth_stencil.is_none() {
        return Err(Refusal::NoAttachment);
    }
    if !key
        .stages
        .iter()
        .any(|s| s.stage == vk::ShaderStageFlags::VERTEX)
    {
        return Err(Refusal::NoVertexStage);
    }
    for (index, stage) in key.stages.iter().enumerate() {
        if key.stages[..index].iter().any(|s| s.stage == stage.stage) {
            return Err(Refusal::DuplicateStage { stage: stage.stage });
        }
    }
    if key.blend.len() != key.compatibility.color.len() {
        return Err(Refusal::BlendAttachmentCount {
            blend: key.blend.len(),
            color: key.compatibility.color.len(),
        });
    }
    match (
        key.depth_stencil.is_some(),
        key.compatibility.depth_stencil.is_some(),
    ) {
        (true, false) => return Err(Refusal::DepthStateWithoutAttachment),
        (false, true) => return Err(Refusal::AttachmentWithoutDepthState),
        _ => {}
    }

    let entries = key
        .stages
        .iter()
        .map(|s| {
            // A guest entry point with an interior NUL cannot be a C string,
            // and the empty name it would truncate to is not this stage's. The
            // shader that named it is refused one layer up, where the name is
            // read; here the fallback is the harmless empty name, which no
            // module exports.
            std::ffi::CString::new(s.entry.as_str())
                .unwrap_or_else(|_| std::ffi::CString::default())
        })
        .collect();
    let bindings = key.bindings.iter().map(|b| b.native()).collect();
    // Only the bindings that need one. A divisor structure listing every
    // binding would name a divisor of one for bindings that have no divisor
    // capability behind them at all.
    let divisors = key
        .bindings
        .iter()
        .filter(|b| b.needs_divisor_structure())
        .map(|b| {
            vk::VertexInputBindingDivisorDescriptionEXT::default()
                .binding(b.binding)
                .divisor(b.divisor)
        })
        .collect();
    let attributes = key.attributes.iter().map(|a| a.native()).collect();
    let attachments = key.blend.iter().map(|a| a.native()).collect();

    let mut dynamic = ALWAYS_DYNAMIC.to_vec();
    dynamic.extend(key.raster.dynamic.states());

    Ok(Build {
        key,
        entries,
        bindings,
        divisors,
        attributes,
        attachments,
        dynamic,
    })
}

impl Build {
    #[must_use]
    pub const fn key(&self) -> &GraphicsKey {
        &self.key
    }

    /// The states this pipeline declares dynamic, in the order it declares
    /// them.
    #[must_use]
    pub fn dynamic_states(&self) -> &[vk::DynamicState] {
        &self.dynamic
    }

    /// Whether any binding needs a divisor structure chained onto the vertex
    /// input state.
    #[must_use]
    pub fn has_divisors(&self) -> bool {
        !self.divisors.is_empty()
    }

    /// Hand `f` the `VkGraphicsPipelineCreateInfo` this build describes, on
    /// `carrier`.
    ///
    /// A closure rather than a returned structure for the reason
    /// [`renderpass::Build::with_render_pass_create_info`] gives: every
    /// sub-state points at an array this value owns, and a returned info would
    /// have to thread a lifetime through all nine of them.
    ///
    /// `render_pass` is used only on [`crate::pass::Carrier::RenderPassObject`]
    /// and must be one this pipeline's [`renderpass::Compatibility`] admits.
    /// On the dynamic-rendering carrier it is ignored and the formats are
    /// chained instead — the same compatibility either way, which is the whole
    /// reason one type serves both.
    pub fn with_create_info<R>(
        &self,
        carrier: crate::pass::Carrier,
        render_pass: vk::RenderPass,
        f: impl FnOnce(&vk::GraphicsPipelineCreateInfo<'_>) -> R,
    ) -> R {
        let stages: Vec<_> = self
            .key
            .stages
            .iter()
            .zip(&self.entries)
            .map(|(stage, entry)| {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(stage.stage)
                    .module(stage.module)
                    .name(entry.as_c_str())
            })
            .collect();

        let mut divisor_state = vk::PipelineVertexInputDivisorStateCreateInfoEXT::default()
            .vertex_binding_divisors(&self.divisors);
        let mut vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&self.bindings)
            .vertex_attribute_descriptions(&self.attributes);
        if !self.divisors.is_empty() {
            vertex_input = vertex_input.push_next(&mut divisor_state);
        }

        let input_assembly = self.input_assembly();
        let rasterization = self.key.raster.native();
        let depth_stencil = self.key.depth_stencil.map(|plan| plan.native());

        // Counts only: the values are dynamic, so the pointers stay null and
        // `vkCmdSetViewport` supplies as many as this declares.
        let viewport = vk::PipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: vk::PipelineViewportStateCreateFlags::empty(),
            viewport_count: self.key.viewports,
            p_viewports: core::ptr::null(),
            scissor_count: self.key.viewports,
            p_scissors: core::ptr::null(),
            _marker: core::marker::PhantomData,
        };

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(self.key.compatibility.samples)
            // Metal has no per-sample fragment execution to request, and
            // asking for it would run the fragment shader once per sample for
            // a guest that priced it once per pixel.
            .sample_shading_enable(false)
            .alpha_to_coverage_enable(self.key.multisample.alpha_to_coverage)
            .alpha_to_one_enable(self.key.multisample.alpha_to_one);

        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            // Metal has no framebuffer logic operation, so there is none to
            // enable and the operation below is never read.
            .logic_op_enable(false)
            .attachments(&self.attachments);

        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&self.dynamic);

        let mut rendering = self.key.compatibility.rendering_info();
        let mut info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(self.key.layout);
        if let Some(state) = depth_stencil.as_ref() {
            info = info.depth_stencil_state(state);
        }
        match carrier {
            crate::pass::Carrier::DynamicRendering => {
                info = info.push_next(&mut rendering);
            }
            crate::pass::Carrier::RenderPassObject => {
                info = info.render_pass(render_pass).subpass(0);
            }
        }
        f(&info)
    }

    /// The input assembly this key's topology means.
    ///
    /// A pipeline built for [`topology::TopologyKey::Class`] or
    /// [`topology::TopologyKey::Any`] declares *a* topology and moves within
    /// what the key allows, so the declared one is the class's list form and
    /// `Any`'s is the triangle list — the type every guest draws most of, and
    /// the one a device that ignores the declaration would be given anyway.
    fn input_assembly(&self) -> vk::PipelineInputAssemblyStateCreateInfo<'static> {
        use reims_vgpu_core::topology::{PrimitiveType, TopologyClass};
        let declared = match self.key.topology {
            topology::TopologyKey::Exact(primitive) => primitive,
            topology::TopologyKey::Class(TopologyClass::Point) => PrimitiveType::Point,
            topology::TopologyKey::Class(TopologyClass::Line) => PrimitiveType::Line,
            topology::TopologyKey::Class(TopologyClass::Triangle) | topology::TopologyKey::Any => {
                PrimitiveType::Triangle
            }
        };
        vk::PipelineInputAssemblyStateCreateInfo {
            topology: topology::topology(declared),
            ..vk::PipelineInputAssemblyStateCreateInfo::default()
        }
    }
}

/// The entry point a Metal-translated shader module exports.
///
/// Every module this rail compiles is emitted with one entry, so the name is a
/// constant rather than a per-shader fact. Named here because both the key and
/// the create info would otherwise spell it separately.
pub const MAIN: &CStr = c"main";

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use reims_vgpu_core::topology::PrimitiveType;
    use std::collections::hash_map::DefaultHasher;
    use std::collections::BTreeSet;
    use std::hash::{Hash, Hasher};

    fn module(raw: u64) -> vk::ShaderModule {
        vk::ShaderModule::from_raw(raw)
    }

    fn stage(flags: vk::ShaderStageFlags, raw: u64) -> StageKey {
        StageKey {
            stage: flags,
            module: module(raw),
            entry: "main".into(),
        }
    }

    fn compatibility(colors: usize, depth: bool) -> renderpass::Compatibility {
        renderpass::Compatibility {
            color: vec![vk::Format::B8G8R8A8_UNORM; colors],
            depth_stencil: depth.then_some(vk::Format::D32_SFLOAT_S8_UINT),
            depth,
            stencil: depth,
            samples: vk::SampleCountFlags::TYPE_1,
        }
    }

    fn depth_state() -> depth_stencil::DepthStencilPlan {
        depth_stencil::DepthStencilPlan {
            depth_test_enable: true,
            depth_write_enable: true,
            depth_compare_op: vk::CompareOp::LESS,
            depth_bounds_test_enable: false,
            stencil_test_enable: false,
            front: depth_stencil::FacePlan::pass_through(),
            back: depth_stencil::FacePlan::pass_through(),
        }
    }

    fn blend_plan() -> blend::AttachmentPlan {
        blend::AttachmentPlan {
            blend_enable: false,
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        }
    }

    fn raster_state() -> raster::RasterizationState {
        raster::plan(
            raster::GuestRasterState::DEFAULT,
            raster::RasterCell::default(),
        )
        .expect("the defaults need no feature")
        .state
    }

    /// One colour attachment, no depth, one vertex and one fragment stage.
    fn key() -> GraphicsKey {
        GraphicsKey {
            stages: vec![
                stage(vk::ShaderStageFlags::VERTEX, 1),
                stage(vk::ShaderStageFlags::FRAGMENT, 2),
            ],
            layout: vk::PipelineLayout::from_raw(9),
            bindings: vec![vertex::BindingPlan {
                binding: 0,
                stride: 16,
                input_rate: vk::VertexInputRate::VERTEX,
                divisor: 1,
            }],
            attributes: vec![vertex::AttributePlan {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 0,
                widened_from: None,
            }],
            topology: topology::TopologyKey::Exact(PrimitiveType::Triangle),
            raster: raster_state(),
            multisample: MultisamplePlan::default(),
            depth_stencil: None,
            blend: vec![blend_plan()],
            compatibility: compatibility(1, false),
            viewports: 1,
        }
    }

    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn the_ordinary_key_builds() {
        let build = build(key()).expect("a vertex stage, one attachment, one viewport");
        assert_eq!(build.key(), &key());
        assert!(!build.has_divisors());
    }

    /// The six states no key carries, because a draw varies them without a
    /// rebuild and all six are encoder commands in Metal.
    #[test]
    fn every_pipeline_declares_the_states_a_draw_may_move() {
        let build = build(key()).expect("built");
        let states: BTreeSet<_> = build.dynamic_states().iter().map(|s| s.as_raw()).collect();
        for wanted in [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::BLEND_CONSTANTS,
            vk::DynamicState::STENCIL_REFERENCE,
            vk::DynamicState::DEPTH_BIAS,
        ] {
            assert!(
                states.contains(&wanted.as_raw()),
                "{wanted:?} is not dynamic"
            );
        }
        assert_eq!(
            states.len(),
            build.dynamic_states().len(),
            "a state is declared twice"
        );
    }

    /// The raster cell's dynamic states join the unconditional ones, and a
    /// host that offers them makes the list longer rather than different.
    #[test]
    fn a_capable_host_adds_to_the_dynamic_list_and_removes_nothing() {
        let bare = build(key()).expect("built");
        let capable_state = raster::plan(
            raster::GuestRasterState::DEFAULT,
            raster::RasterCell {
                depth_clamp: true,
                fill_mode_non_solid: true,
                dynamic_cull_and_winding: true,
                dynamic_polygon_mode: true,
                dynamic_depth_clamp: true,
            },
        )
        .expect("capable")
        .state;
        let capable = build(GraphicsKey {
            raster: capable_state,
            ..key()
        })
        .expect("built");

        let bare_set: BTreeSet<_> = bare.dynamic_states().iter().map(|s| s.as_raw()).collect();
        let capable_set: BTreeSet<_> = capable
            .dynamic_states()
            .iter()
            .map(|s| s.as_raw())
            .collect();
        assert!(bare_set.is_subset(&capable_set));
        assert_eq!(
            capable_set.len(),
            bare_set.len() + 4,
            "cull, front, polygon, clamp"
        );
    }

    /// A pipeline with nothing to run is refused rather than created and never
    /// drawn with.
    #[test]
    fn a_pipeline_needs_a_vertex_stage_and_may_not_declare_one_stage_twice() {
        assert_eq!(
            build(GraphicsKey {
                stages: vec![stage(vk::ShaderStageFlags::FRAGMENT, 2)],
                ..key()
            })
            .unwrap_err(),
            Refusal::NoVertexStage
        );
        assert_eq!(
            build(GraphicsKey {
                stages: vec![
                    stage(vk::ShaderStageFlags::VERTEX, 1),
                    stage(vk::ShaderStageFlags::VERTEX, 3),
                ],
                ..key()
            })
            .unwrap_err(),
            Refusal::DuplicateStage {
                stage: vk::ShaderStageFlags::VERTEX
            }
        );
        // A vertex stage alone is legal: a depth-only pass has no fragment
        // shader and still rasterizes.
        assert!(build(GraphicsKey {
            stages: vec![stage(vk::ShaderStageFlags::VERTEX, 1)],
            ..key()
        })
        .is_ok());
    }

    /// The blend list and the pass's colour attachments are the same count, in
    /// both directions.
    #[test]
    fn a_blend_attachment_count_that_is_not_the_passs_is_refused_either_way() {
        assert_eq!(
            build(GraphicsKey {
                blend: vec![],
                ..key()
            })
            .unwrap_err(),
            Refusal::BlendAttachmentCount { blend: 0, color: 1 }
        );
        assert_eq!(
            build(GraphicsKey {
                blend: vec![blend_plan(), blend_plan()],
                ..key()
            })
            .unwrap_err(),
            Refusal::BlendAttachmentCount { blend: 2, color: 1 }
        );
        // And the matching pair for a multiple-target pass builds.
        assert!(build(GraphicsKey {
            blend: vec![blend_plan(), blend_plan()],
            compatibility: compatibility(2, false),
            ..key()
        })
        .is_ok());
    }

    /// A depth state and a depth attachment arrive together or not at all.
    /// Either half alone is a piece of guest state that silently does nothing.
    #[test]
    fn a_depth_state_and_a_depth_attachment_are_refused_apart() {
        let state = depth_stencil::DepthStencilPlan {
            depth_test_enable: true,
            depth_write_enable: true,
            depth_compare_op: vk::CompareOp::LESS,
            depth_bounds_test_enable: false,
            stencil_test_enable: false,
            front: depth_stencil::FacePlan::pass_through(),
            back: depth_stencil::FacePlan::pass_through(),
        };
        assert_eq!(
            build(GraphicsKey {
                depth_stencil: Some(state),
                ..key()
            })
            .unwrap_err(),
            Refusal::DepthStateWithoutAttachment
        );
        assert_eq!(
            build(GraphicsKey {
                compatibility: compatibility(1, true),
                ..key()
            })
            .unwrap_err(),
            Refusal::AttachmentWithoutDepthState
        );
        assert!(build(GraphicsKey {
            depth_stencil: Some(state),
            compatibility: compatibility(1, true),
            ..key()
        })
        .is_ok());
    }

    /// A pass with nothing attached and a pipeline with no viewport are both
    /// draws that produce nothing.
    #[test]
    fn a_pipeline_that_could_write_nowhere_is_refused() {
        assert_eq!(
            build(GraphicsKey {
                viewports: 0,
                ..key()
            })
            .unwrap_err(),
            Refusal::NoViewport
        );
        assert_eq!(
            build(GraphicsKey {
                blend: vec![],
                compatibility: compatibility(0, false),
                ..key()
            })
            .unwrap_err(),
            Refusal::NoAttachment
        );
        // A depth-only pass has no colour attachment and is not nothing.
        assert!(build(GraphicsKey {
            blend: vec![],
            depth_stencil: Some(depth_state()),
            compatibility: compatibility(0, true),
            ..key()
        })
        .is_ok());
    }

    /// Only the bindings that need one appear in the divisor structure. A
    /// structure listing every binding would name a divisor for bindings with
    /// no divisor capability behind them.
    #[test]
    fn the_divisor_structure_names_only_the_bindings_that_have_one() {
        let plain = vertex::BindingPlan {
            binding: 0,
            stride: 16,
            input_rate: vk::VertexInputRate::VERTEX,
            divisor: 1,
        };
        assert!(!build(GraphicsKey {
            bindings: vec![plain],
            ..key()
        })
        .expect("built")
        .has_divisors());

        for divisor in [0, 2, 7] {
            let build = build(GraphicsKey {
                bindings: vec![
                    plain,
                    vertex::BindingPlan {
                        binding: 1,
                        input_rate: vk::VertexInputRate::INSTANCE,
                        divisor,
                        ..plain
                    },
                ],
                ..key()
            })
            .expect("built");
            assert!(build.has_divisors(), "divisor {divisor}");
            assert_eq!(build.divisors.len(), 1, "only the second binding");
            assert_eq!(build.divisors[0].binding, 1);
            assert_eq!(build.divisors[0].divisor, divisor);
        }
    }

    /// The key is the cache dimension, so every part of it must move the hash.
    /// A field that did not would let one compilation serve a draw it cannot.
    #[test]
    fn every_part_of_the_key_moves_the_hash() {
        let base = key();
        let variants: Vec<(&str, GraphicsKey)> = vec![
            (
                "stages",
                GraphicsKey {
                    stages: vec![stage(vk::ShaderStageFlags::VERTEX, 99)],
                    ..base.clone()
                },
            ),
            (
                "layout",
                GraphicsKey {
                    layout: vk::PipelineLayout::from_raw(10),
                    ..base.clone()
                },
            ),
            (
                "bindings",
                GraphicsKey {
                    bindings: vec![vertex::BindingPlan {
                        stride: 32,
                        ..base.bindings[0]
                    }],
                    ..base.clone()
                },
            ),
            (
                "attributes",
                GraphicsKey {
                    attributes: vec![vertex::AttributePlan {
                        offset: 4,
                        ..base.attributes[0]
                    }],
                    ..base.clone()
                },
            ),
            (
                "topology",
                GraphicsKey {
                    topology: topology::TopologyKey::Any,
                    ..base.clone()
                },
            ),
            (
                "raster",
                GraphicsKey {
                    raster: raster::RasterizationState {
                        cull_mode: vk::CullModeFlags::BACK,
                        ..base.raster
                    },
                    ..base.clone()
                },
            ),
            (
                "multisample",
                GraphicsKey {
                    multisample: MultisamplePlan {
                        alpha_to_coverage: true,
                        alpha_to_one: false,
                    },
                    ..base.clone()
                },
            ),
            (
                "blend",
                GraphicsKey {
                    blend: vec![blend::AttachmentPlan {
                        blend_enable: true,
                        ..blend_plan()
                    }],
                    ..base.clone()
                },
            ),
            (
                "compatibility",
                GraphicsKey {
                    compatibility: renderpass::Compatibility {
                        samples: vk::SampleCountFlags::TYPE_4,
                        ..compatibility(1, false)
                    },
                    ..base.clone()
                },
            ),
            (
                "viewports",
                GraphicsKey {
                    viewports: 2,
                    ..base.clone()
                },
            ),
        ];
        let mut seen = BTreeSet::new();
        seen.insert(hash_of(&base));
        for (name, variant) in variants {
            assert_ne!(variant, base, "{name} did not change the key");
            assert!(
                seen.insert(hash_of(&variant)),
                "{name} collides with an earlier key"
            );
        }
        // The depth-stencil dimension needs a pass that has one, so it is
        // varied against its own base rather than the colour-only one above.
        let with_depth = GraphicsKey {
            depth_stencil: Some(depth_state()),
            compatibility: compatibility(1, true),
            ..base.clone()
        };
        let written = GraphicsKey {
            depth_stencil: with_depth
                .depth_stencil
                .map(|plan| depth_stencil::DepthStencilPlan {
                    depth_write_enable: !plan.depth_write_enable,
                    ..plan
                }),
            ..with_depth.clone()
        };
        assert_ne!(hash_of(&with_depth), hash_of(&written));
    }

    /// The two carriers differ in how the pass is named and in nothing else.
    /// A key that served one and not the other would make a host that took the
    /// other rung compile a different number of pipelines for one guest.
    #[test]
    fn both_carriers_take_the_same_key() {
        let build = build(key()).expect("built");
        for carrier in [
            crate::pass::Carrier::DynamicRendering,
            crate::pass::Carrier::RenderPassObject,
        ] {
            let (render_pass, subpass, chained) =
                build.with_create_info(carrier, vk::RenderPass::from_raw(5), |info| {
                    (info.render_pass, info.subpass, !info.p_next.is_null())
                });
            match carrier {
                crate::pass::Carrier::DynamicRendering => {
                    assert_eq!(render_pass, vk::RenderPass::null());
                    assert!(chained, "the formats are chained");
                }
                crate::pass::Carrier::RenderPassObject => {
                    assert_eq!(render_pass, vk::RenderPass::from_raw(5));
                    assert_eq!(subpass, 0);
                }
            }
        }
    }

    /// The create info points at this build's arrays and carries its counts.
    #[test]
    fn the_create_info_is_the_build() {
        let build = build(GraphicsKey {
            viewports: 3,
            ..key()
        })
        .expect("built");
        build.with_create_info(
            crate::pass::Carrier::DynamicRendering,
            vk::RenderPass::null(),
            |info| {
                assert_eq!(info.stage_count, 2);
                assert_eq!(info.layout, vk::PipelineLayout::from_raw(9));
                let viewport = unsafe { &*info.p_viewport_state };
                assert_eq!(viewport.viewport_count, 3);
                assert_eq!(viewport.scissor_count, 3);
                // Dynamic, so the pipeline names counts and no values.
                assert!(viewport.p_viewports.is_null());
                assert!(viewport.p_scissors.is_null());

                let vertex_input = unsafe { &*info.p_vertex_input_state };
                assert_eq!(vertex_input.vertex_binding_description_count, 1);
                assert_eq!(vertex_input.vertex_attribute_description_count, 1);
                assert!(vertex_input.p_next.is_null(), "no divisor structure");

                let blend = unsafe { &*info.p_color_blend_state };
                assert_eq!(blend.attachment_count, 1);
                // Metal has no framebuffer logic operation.
                assert_eq!(blend.logic_op_enable, vk::FALSE);

                let multisample = unsafe { &*info.p_multisample_state };
                assert_eq!(
                    multisample.rasterization_samples,
                    vk::SampleCountFlags::TYPE_1
                );
                assert_eq!(multisample.sample_shading_enable, vk::FALSE);

                // No depth-stencil attachment, so no state at all rather than
                // a disabled one.
                assert!(info.p_depth_stencil_state.is_null());

                let dynamic = unsafe { &*info.p_dynamic_state };
                assert_eq!(dynamic.dynamic_state_count as usize, build.dynamic.len());
            },
        );
    }

    /// A key whose bindings need divisors chains the structure the divisor
    /// capability is spent through.
    #[test]
    fn a_divisor_binding_chains_the_structure_onto_the_vertex_input_state() {
        let build = build(GraphicsKey {
            bindings: vec![vertex::BindingPlan {
                binding: 0,
                stride: 16,
                input_rate: vk::VertexInputRate::INSTANCE,
                divisor: 4,
            }],
            ..key()
        })
        .expect("built");
        build.with_create_info(
            crate::pass::Carrier::DynamicRendering,
            vk::RenderPass::null(),
            |info| {
                let vertex_input = unsafe { &*info.p_vertex_input_state };
                assert!(!vertex_input.p_next.is_null(), "the divisor structure");
            },
        );
    }

    /// A pipeline built for a topology class declares a member of that class,
    /// and one built for any topology declares the triangle list.
    #[test]
    fn a_class_pipeline_declares_a_topology_its_key_admits() {
        use reims_vgpu_core::topology::TopologyClass;
        for (key_topology, expected) in [
            (
                topology::TopologyKey::Exact(PrimitiveType::LineStrip),
                vk::PrimitiveTopology::LINE_STRIP,
            ),
            (
                topology::TopologyKey::Class(TopologyClass::Line),
                vk::PrimitiveTopology::LINE_LIST,
            ),
            (
                topology::TopologyKey::Class(TopologyClass::Point),
                vk::PrimitiveTopology::POINT_LIST,
            ),
            (
                topology::TopologyKey::Class(TopologyClass::Triangle),
                vk::PrimitiveTopology::TRIANGLE_LIST,
            ),
            (
                topology::TopologyKey::Any,
                vk::PrimitiveTopology::TRIANGLE_LIST,
            ),
        ] {
            let build = build(GraphicsKey {
                topology: key_topology,
                ..key()
            })
            .expect("built");
            assert_eq!(
                build.input_assembly().topology,
                expected,
                "{key_topology:?}"
            );
            // Never restarting: Metal has no primitive-restart index and a
            // guest's largest index would silently cut its strip.
            assert_eq!(
                build.input_assembly().primitive_restart_enable,
                vk::FALSE,
                "{key_topology:?}"
            );
        }
    }

    /// An entry point with an interior NUL cannot become a C string. It falls
    /// back to a name no module exports rather than to the truncation, which
    /// would silently name some other entry.
    #[test]
    fn an_unrepresentable_entry_point_does_not_truncate_onto_another() {
        let build = build(GraphicsKey {
            stages: vec![StageKey {
                entry: "main\0evil".into(),
                ..stage(vk::ShaderStageFlags::VERTEX, 1)
            }],
            ..key()
        })
        .expect("built");
        assert_eq!(build.entries[0].as_c_str(), c"");
        assert_ne!(build.entries[0].as_c_str(), MAIN);
    }
}
