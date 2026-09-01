//! Turning a declared barrier into stage and access masks, and refusing the
//! stages this host has no equivalent for.
//!
//! # Why the translation is here and the barrier is not
//!
//! Every barrier opcode on the guest's wire is a proven no-op *for a particular
//! executor's submission granularity* — one host submission per pass boundary,
//! one per dispatch. [`reims_vgpu_core::sync`] therefore keeps the barrier as an
//! operation with its declared scope and stages intact, because the replacement
//! exists in order to batch differently and a merged batch is precisely where
//! the no-op stops being one.
//!
//! Which masks that becomes, and whether this host can express the stages the
//! guest named, is a host question. So it is here, and nothing above this crate
//! decides it.
//!
//! # The plan is `synchronization2`, and the legacy form is a mapping of it
//!
//! Vulkan 1.2 is the baseline and `VK_KHR_synchronization2` is an extension
//! there, so a host may need the older `vkCmdPipelineBarrier` masks. The rule
//! the plan sets is that a legacy emitter may exist only if it consumes the
//! *same* plan — so [`BarrierPlan`] is the one answer and
//! [`BarrierPlan::legacy`] maps it, rather than translating the guest's request
//! a second time. Two translations would be two readings of one request, and the
//! host on the older path would be the one nobody tested.
//!
//! The mapping is not a truncation. `synchronization2` split the old
//! `SHADER_READ` into sampled and storage reads and gave them bits above the
//! 32-bit space, so the older form is genuinely coarser and the map says by how
//! much in one table. [`BarrierPlan::unmapped_bits`] is the check that the table
//! covers everything this module emits — a bit with no older equivalent would
//! otherwise be silently dropped, and the legacy host would be asked for less
//! ordering than the guest requested.
//!
//! # A stage this host cannot express is refused by name
//!
//! Metal's tile stage has no core Vulkan equivalent, and its object and mesh
//! stages need `VK_EXT_mesh_shader`. Folding either into the fragment stage
//! would execute a barrier the guest did not ask for and silently drop the
//! ordering it did — so they are [`Decline`]s, and the mesh case is gated on the
//! capability being present rather than on a device or driver name.
//!
//! Undeclared bits are refused for the same reason
//! [`reims_vgpu_core::sync`]'s vocabulary keeps them representable: masking a
//! bit down to its declared neighbours reports a guest request the device is
//! sure it understood.
//!
//! # What this does not do
//!
//! A barrier over a listed set of resources becomes one image or buffer memory
//! barrier per resource, and this cannot name a resource — the list is a window
//! into a transaction's own arena. So the masks come back and the caller applies
//! them to the resources it resolved. The masks are the part that is a host
//! policy decision; which resources they cover is not.

use ash::vk;
use reims_vgpu_core::sync::RenderStages;
use reims_vgpu_core::sync::{BarrierOp, BarrierScope, BarrierTarget};

/// What this host can express, as capabilities rather than names.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageSupport {
    /// `VK_EXT_mesh_shader` is present and enabled, so the object and mesh
    /// stages have equivalents.
    pub mesh_shader: bool,
}

/// Why a barrier could not be translated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decline {
    /// The scope named bits the API does not declare.
    UndeclaredScopeBits { bits: u32 },
    /// The stage mask named bits the API does not declare.
    UndeclaredStageBits { bits: u32 },
    /// The guest named the tile stage. Metal's tile shading has no core Vulkan
    /// equivalent, and the imageblock memory it orders is not a thing this rail
    /// has to order.
    TileStage,
    /// The guest named the object or mesh stage and this host has no mesh
    /// shading. A capability answer, not a device one.
    MeshStagesUnavailable { stages: u32 },
}

impl Decline {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::UndeclaredScopeBits { .. } => "vk_barrier_undeclared_scope_bits",
            Self::UndeclaredStageBits { .. } => "vk_barrier_undeclared_stage_bits",
            Self::TileStage => "vk_barrier_tile_stage",
            Self::MeshStagesUnavailable { .. } => "vk_barrier_mesh_stages_unavailable",
        }
    }
}

impl std::fmt::Display for Decline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::UndeclaredScopeBits { bits } | Self::UndeclaredStageBits { bits } => {
                write!(f, "{} bits={bits:#x}", self.slug())
            }
            Self::TileStage => write!(f, "{}", self.slug()),
            Self::MeshStagesUnavailable { stages } => {
                write!(f, "{} stages={stages:#x}", self.slug())
            }
        }
    }
}

/// The one answer: what has to finish, and what may then proceed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BarrierPlan {
    pub src_stages: vk::PipelineStageFlags2,
    pub dst_stages: vk::PipelineStageFlags2,
    pub src_access: vk::AccessFlags2,
    pub dst_access: vk::AccessFlags2,
}

/// The same plan in the pre-`synchronization2` flag types.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LegacyBarrier {
    pub src_stages: vk::PipelineStageFlags,
    pub dst_stages: vk::PipelineStageFlags,
    pub src_access: vk::AccessFlags,
    pub dst_access: vk::AccessFlags,
}

impl BarrierPlan {
    /// Whether this plan orders anything at all.
    ///
    /// A guest may legally send a barrier over an empty scope. That orders
    /// nothing, and saying so here keeps "the guest asked for nothing" apart
    /// from "the device decided it owed nothing" — the two answers a barrier
    /// census has to distinguish.
    #[must_use]
    pub fn orders_anything(self) -> bool {
        !self.src_stages.is_empty() || !self.dst_stages.is_empty()
    }

    /// Map onto the pre-`synchronization2` flags.
    ///
    /// Coarser by construction: the older vocabulary has one `SHADER_READ`
    /// where the newer has a sampled read and a storage read. The map is the one
    /// place that coarsening happens, and [`BarrierPlan::unmapped_bits`] is what
    /// says it lost nothing rather than a comment claiming it.
    #[must_use]
    pub fn legacy(self) -> LegacyBarrier {
        LegacyBarrier {
            src_stages: map_stages(self.src_stages),
            dst_stages: map_stages(self.dst_stages),
            src_access: map_access(self.src_access),
            dst_access: map_access(self.dst_access),
        }
    }

    /// The bits with no entry in the legacy map, or empty.
    ///
    /// Non-empty would mean this module had started emitting a
    /// `synchronization2` value the map does not cover, and a host on the older
    /// path would then be asked for less ordering than the guest requested. The
    /// answer is to extend the map or refuse the host, never to drop the bit.
    #[must_use]
    pub fn unmapped_bits(self) -> (u64, u64) {
        let stages = (self.src_stages | self.dst_stages).as_raw() & !mapped_stage_bits();
        let access = (self.src_access | self.dst_access).as_raw() & !mapped_access_bits();
        (stages, access)
    }
}

/// Every `synchronization2` stage this module emits, and its older equivalent.
const STAGE_MAP: &[(vk::PipelineStageFlags2, vk::PipelineStageFlags)] = &[
    (
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    ),
    (
        vk::PipelineStageFlags2::VERTEX_SHADER,
        vk::PipelineStageFlags::VERTEX_SHADER,
    ),
    (
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
    ),
    (
        vk::PipelineStageFlags2::TASK_SHADER_EXT,
        vk::PipelineStageFlags::TASK_SHADER_EXT,
    ),
    (
        vk::PipelineStageFlags2::MESH_SHADER_EXT,
        vk::PipelineStageFlags::MESH_SHADER_EXT,
    ),
];

/// Every `synchronization2` access this module emits, and its older equivalent.
///
/// Two rows collapse onto `SHADER_READ`: that is the coarsening the older
/// vocabulary forces, and it is stated here rather than discovered.
const ACCESS_MAP: &[(vk::AccessFlags2, vk::AccessFlags)] = &[
    (
        vk::AccessFlags2::UNIFORM_READ,
        vk::AccessFlags::UNIFORM_READ,
    ),
    (
        vk::AccessFlags2::SHADER_SAMPLED_READ,
        vk::AccessFlags::SHADER_READ,
    ),
    (
        vk::AccessFlags2::SHADER_STORAGE_READ,
        vk::AccessFlags::SHADER_READ,
    ),
    (
        vk::AccessFlags2::SHADER_STORAGE_WRITE,
        vk::AccessFlags::SHADER_WRITE,
    ),
    (
        vk::AccessFlags2::INPUT_ATTACHMENT_READ,
        vk::AccessFlags::INPUT_ATTACHMENT_READ,
    ),
    (
        vk::AccessFlags2::COLOR_ATTACHMENT_READ,
        vk::AccessFlags::COLOR_ATTACHMENT_READ,
    ),
    (
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
    ),
    (
        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
    ),
    (
        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
    ),
    (vk::AccessFlags2::MEMORY_READ, vk::AccessFlags::MEMORY_READ),
    (
        vk::AccessFlags2::MEMORY_WRITE,
        vk::AccessFlags::MEMORY_WRITE,
    ),
];

fn mapped_stage_bits() -> u64 {
    STAGE_MAP.iter().fold(0, |acc, (new, _)| acc | new.as_raw())
}

fn mapped_access_bits() -> u64 {
    ACCESS_MAP
        .iter()
        .fold(0, |acc, (new, _)| acc | new.as_raw())
}

fn map_stages(stages: vk::PipelineStageFlags2) -> vk::PipelineStageFlags {
    STAGE_MAP
        .iter()
        .filter(|(new, _)| stages.contains(*new))
        .fold(vk::PipelineStageFlags::empty(), |acc, (_, old)| acc | *old)
}

fn map_access(access: vk::AccessFlags2) -> vk::AccessFlags {
    ACCESS_MAP
        .iter()
        .filter(|(new, _)| access.contains(*new))
        .fold(vk::AccessFlags::empty(), |acc, (_, old)| acc | *old)
}

/// Translate one declared barrier.
///
/// # Errors
///
/// If the record named an undeclared bit, or a stage this host cannot express.
pub fn translate(op: &BarrierOp, support: StageSupport) -> Result<BarrierPlan, Decline> {
    let src_stages = stages(op.after_stages, support)?;
    let dst_stages = stages(op.before_stages, support)?;
    let (src_access, dst_access) = match op.target {
        // The record named a scope of memory. Both halves get the scope's
        // access, because the barrier is an ordering of that memory against
        // itself: the guest wrote it through one stage set and reads it through
        // the other.
        BarrierTarget::Scope(scope) => {
            let bits = scope.undeclared_bits();
            if bits != 0 {
                return Err(Decline::UndeclaredScopeBits { bits });
            }
            let access = scope_access(scope);
            (access, access)
        }
        // The record named resources and no usage, so the direction is not
        // established. Conservative both ways: an ordering that is too strong
        // costs throughput, and one that is too weak is a race.
        BarrierTarget::Resources(_) => (
            vk::AccessFlags2::MEMORY_WRITE,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        ),
        // `textureBarrier` makes previously written fragments readable by later
        // ones in the same pass. That is exactly colour-attachment write
        // becoming shader read; the attachments are the pass's and not the
        // record's, which is why the caller supplies them.
        BarrierTarget::Texture => (
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags2::SHADER_SAMPLED_READ | vk::AccessFlags2::INPUT_ATTACHMENT_READ,
        ),
    };
    // A barrier over nothing orders nothing, and the plan says so by being
    // empty rather than by carrying stages with no access.
    if !op.orders_anything() {
        return Ok(BarrierPlan::default());
    }
    Ok(BarrierPlan {
        src_stages,
        dst_stages,
        src_access,
        dst_access,
    })
}

/// The access mask one declared scope names.
fn scope_access(scope: BarrierScope) -> vk::AccessFlags2 {
    let mut access = vk::AccessFlags2::empty();
    if scope.0 & BarrierScope::BUFFERS != 0 {
        access |= vk::AccessFlags2::SHADER_STORAGE_READ
            | vk::AccessFlags2::SHADER_STORAGE_WRITE
            | vk::AccessFlags2::UNIFORM_READ;
    }
    if scope.0 & BarrierScope::TEXTURES != 0 {
        access |= vk::AccessFlags2::SHADER_SAMPLED_READ
            | vk::AccessFlags2::SHADER_STORAGE_READ
            | vk::AccessFlags2::SHADER_STORAGE_WRITE;
    }
    if scope.0 & BarrierScope::RENDER_TARGETS != 0 {
        access |= vk::AccessFlags2::COLOR_ATTACHMENT_READ
            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
    }
    access
}

/// The pipeline stages one declared stage mask names.
///
/// `None` is the compute rail: its barrier selectors carry no stage argument at
/// all, so the stage is the one its records execute in rather than an empty
/// mask. An empty mask would be a barrier that orders nothing, which is not what
/// a compute barrier is.
fn stages(
    declared: Option<RenderStages>,
    support: StageSupport,
) -> Result<vk::PipelineStageFlags2, Decline> {
    let Some(declared) = declared else {
        return Ok(vk::PipelineStageFlags2::COMPUTE_SHADER);
    };
    let bits = declared.undeclared_bits();
    if bits != 0 {
        return Err(Decline::UndeclaredStageBits { bits });
    }
    if declared.0 & RenderStages::TILE != 0 {
        return Err(Decline::TileStage);
    }
    let mesh = declared.0 & (RenderStages::OBJECT | RenderStages::MESH);
    if mesh != 0 && !support.mesh_shader {
        return Err(Decline::MeshStagesUnavailable { stages: mesh });
    }
    let mut out = vk::PipelineStageFlags2::empty();
    if declared.0 & RenderStages::VERTEX != 0 {
        out |= vk::PipelineStageFlags2::VERTEX_SHADER;
    }
    if declared.0 & RenderStages::FRAGMENT != 0 {
        out |= vk::PipelineStageFlags2::FRAGMENT_SHADER;
    }
    if declared.0 & RenderStages::OBJECT != 0 {
        out |= vk::PipelineStageFlags2::TASK_SHADER_EXT;
    }
    if declared.0 & RenderStages::MESH != 0 {
        out |= vk::PipelineStageFlags2::MESH_SHADER_EXT;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::sync::ResourceSpan;

    fn scope(bits: u32, after: Option<u32>, before: Option<u32>) -> BarrierOp {
        BarrierOp {
            target: BarrierTarget::Scope(BarrierScope(bits)),
            after_stages: after.map(RenderStages),
            before_stages: before.map(RenderStages),
        }
    }

    const RENDER: StageSupport = StageSupport { mesh_shader: false };
    const MESH: StageSupport = StageSupport { mesh_shader: true };

    /// A compute barrier's selectors carry no stages, and the compute stage is
    /// what its records execute in — not an empty mask, which would order
    /// nothing.
    #[test]
    fn a_stageless_barrier_is_the_compute_stage_and_not_an_empty_mask() {
        let plan = translate(
            &scope(BarrierScope::BUFFERS | BarrierScope::TEXTURES, None, None),
            RENDER,
        )
        .expect("declared bits only");
        assert_eq!(plan.src_stages, vk::PipelineStageFlags2::COMPUTE_SHADER);
        assert_eq!(plan.dst_stages, vk::PipelineStageFlags2::COMPUTE_SHADER);
        assert!(plan.orders_anything());
        assert!(plan
            .src_access
            .contains(vk::AccessFlags2::SHADER_STORAGE_WRITE));
        assert!(plan
            .dst_access
            .contains(vk::AccessFlags2::SHADER_SAMPLED_READ));
    }

    #[test]
    fn a_render_barrier_carries_the_stages_the_record_declared() {
        let plan = translate(
            &scope(
                BarrierScope::RENDER_TARGETS,
                Some(RenderStages::FRAGMENT),
                Some(RenderStages::VERTEX | RenderStages::FRAGMENT),
            ),
            RENDER,
        )
        .expect("declared bits only");
        assert_eq!(plan.src_stages, vk::PipelineStageFlags2::FRAGMENT_SHADER);
        assert_eq!(
            plan.dst_stages,
            vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER
        );
        assert!(plan
            .src_access
            .contains(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE));
    }

    /// A barrier over nothing is legal to send and orders nothing. It must not
    /// come back as stages with no access, which a caller would emit.
    #[test]
    fn an_empty_scope_orders_nothing() {
        let plan = translate(&scope(0, Some(RenderStages::FRAGMENT), None), RENDER)
            .expect("zero is a declared value");
        assert_eq!(plan, BarrierPlan::default());
        assert!(!plan.orders_anything());
    }

    #[test]
    fn an_empty_resource_list_orders_nothing() {
        let op = BarrierOp {
            target: BarrierTarget::Resources(ResourceSpan { start: 0, len: 0 }),
            after_stages: Some(RenderStages(RenderStages::FRAGMENT)),
            before_stages: Some(RenderStages(RenderStages::VERTEX)),
        };
        assert_eq!(translate(&op, RENDER), Ok(BarrierPlan::default()));
    }

    /// The record names resources and no usage, so the direction is not
    /// established. Too strong costs throughput; too weak is a race.
    #[test]
    fn a_resource_list_with_no_usage_is_ordered_conservatively() {
        let op = BarrierOp {
            target: BarrierTarget::Resources(ResourceSpan { start: 4, len: 3 }),
            after_stages: None,
            before_stages: None,
        };
        let plan = translate(&op, RENDER).expect("no stage bits to be wrong");
        assert_eq!(plan.src_access, vk::AccessFlags2::MEMORY_WRITE);
        assert_eq!(
            plan.dst_access,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE
        );
    }

    /// Masking a bit down to its declared neighbours reports a guest request
    /// the device is sure it understood.
    #[test]
    fn an_undeclared_bit_is_refused_and_not_masked() {
        assert_eq!(
            translate(&scope(BarrierScope::BUFFERS | 0x80, None, None), RENDER),
            Err(Decline::UndeclaredScopeBits { bits: 0x80 })
        );
        assert_eq!(
            translate(&scope(BarrierScope::BUFFERS, Some(0x40), None), RENDER),
            Err(Decline::UndeclaredStageBits { bits: 0x40 })
        );
    }

    /// Folding the tile stage into the fragment stage would execute a barrier
    /// the guest did not ask for and drop the ordering it did.
    #[test]
    fn the_tile_stage_is_refused_by_name() {
        assert_eq!(
            translate(
                &scope(
                    BarrierScope::TEXTURES,
                    Some(RenderStages::FRAGMENT | RenderStages::TILE),
                    Some(RenderStages::FRAGMENT)
                ),
                MESH
            ),
            Err(Decline::TileStage),
            "and mesh support does not make a tile stage expressible"
        );
    }

    /// Gated on the capability, not on a device.
    #[test]
    fn the_mesh_stages_follow_the_capability() {
        let op = scope(
            BarrierScope::TEXTURES,
            Some(RenderStages::OBJECT | RenderStages::MESH),
            Some(RenderStages::FRAGMENT),
        );
        assert_eq!(
            translate(&op, RENDER),
            Err(Decline::MeshStagesUnavailable {
                stages: RenderStages::OBJECT | RenderStages::MESH
            })
        );
        let plan = translate(&op, MESH).expect("the capability is present");
        assert_eq!(
            plan.src_stages,
            vk::PipelineStageFlags2::TASK_SHADER_EXT | vk::PipelineStageFlags2::MESH_SHADER_EXT
        );
    }

    /// The legacy path consumes the same plan. A map that dropped a bit would
    /// ask the host on the older path for less ordering than the guest wrote.
    #[test]
    fn the_legacy_map_covers_everything_this_module_emits() {
        let cases = [
            scope(BarrierScope::BUFFERS, None, None),
            scope(BarrierScope::TEXTURES, None, None),
            scope(
                BarrierScope::RENDER_TARGETS,
                Some(RenderStages::FRAGMENT),
                Some(RenderStages::VERTEX),
            ),
            scope(
                BarrierScope::BUFFERS | BarrierScope::TEXTURES | BarrierScope::RENDER_TARGETS,
                Some(RenderStages::VERTEX | RenderStages::FRAGMENT),
                Some(RenderStages::FRAGMENT),
            ),
            BarrierOp {
                target: BarrierTarget::Texture,
                after_stages: Some(RenderStages(RenderStages::FRAGMENT)),
                before_stages: Some(RenderStages(RenderStages::FRAGMENT)),
            },
            BarrierOp {
                target: BarrierTarget::Resources(ResourceSpan { start: 0, len: 1 }),
                after_stages: None,
                before_stages: None,
            },
            scope(
                BarrierScope::TEXTURES,
                Some(RenderStages::OBJECT | RenderStages::MESH),
                Some(RenderStages::FRAGMENT),
            ),
        ];
        for op in cases {
            let plan = translate(&op, MESH).expect("declared bits only");
            assert_eq!(
                plan.unmapped_bits(),
                (0, 0),
                "a bit with no older equivalent reached the plan for {op:?}"
            );
        }
    }

    /// The coarsening the older vocabulary forces, stated rather than
    /// discovered: two distinct reads become one.
    #[test]
    fn the_two_shader_reads_become_one_in_the_older_vocabulary() {
        let plan = translate(&scope(BarrierScope::TEXTURES, None, None), RENDER)
            .expect("declared bits only");
        assert!(plan
            .src_access
            .contains(vk::AccessFlags2::SHADER_SAMPLED_READ));
        assert!(plan
            .src_access
            .contains(vk::AccessFlags2::SHADER_STORAGE_READ));
        let legacy = plan.legacy();
        assert!(legacy.src_access.contains(vk::AccessFlags::SHADER_READ));
        assert!(legacy.src_access.contains(vk::AccessFlags::SHADER_WRITE));
        assert_eq!(
            legacy.src_stages,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            "and the stage maps one-to-one"
        );
    }

    /// A mesh-stage barrier maps onto the extension's own older stage bits, so
    /// a host with mesh shading and no `synchronization2` is still served.
    #[test]
    fn the_mesh_stages_have_an_older_equivalent() {
        let plan = translate(
            &scope(
                BarrierScope::TEXTURES,
                Some(RenderStages::MESH),
                Some(RenderStages::FRAGMENT),
            ),
            MESH,
        )
        .expect("the capability is present");
        assert_eq!(plan.unmapped_bits(), (0, 0));
        assert_eq!(
            plan.legacy().src_stages,
            vk::PipelineStageFlags::MESH_SHADER_EXT
        );
    }

    #[test]
    fn a_texture_barrier_orders_the_passes_own_attachments() {
        let plan = translate(
            &BarrierOp {
                target: BarrierTarget::Texture,
                after_stages: Some(RenderStages(RenderStages::FRAGMENT)),
                before_stages: Some(RenderStages(RenderStages::FRAGMENT)),
            },
            RENDER,
        )
        .expect("declared bits only");
        assert_eq!(plan.src_access, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE);
        assert!(plan
            .dst_access
            .contains(vk::AccessFlags2::SHADER_SAMPLED_READ));
        assert!(
            plan.orders_anything(),
            "it names nothing and orders its pass"
        );
    }

    #[test]
    fn every_decline_has_its_own_slug() {
        let all = [
            Decline::UndeclaredScopeBits { bits: 1 },
            Decline::UndeclaredStageBits { bits: 1 },
            Decline::TileStage,
            Decline::MeshStagesUnavailable { stages: 1 },
        ];
        let mut slugs: Vec<&str> = all.iter().map(|d| d.slug()).collect();
        slugs.sort_unstable();
        let count = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), count);
        assert!(Decline::TileStage.to_string().contains("tile_stage"));
    }
}
