//! A checked depth-stencil state as `VkPipelineDepthStencilStateCreateInfo`
//! would hold it.
//!
//! # Metal has no depth-test enable, and Vulkan's gates the write
//!
//! `MTLDepthStencilDescriptor` carries a compare function and a write flag.
//! Depth is always tested; "no depth test" is `Always` with writes off.
//! Vulkan's `depthTestEnable` is not the same switch — with it clear, Vulkan
//! performs no test *and no write*, whatever `depthWriteEnable` says.
//!
//! So this rail sets `depthTestEnable` unconditionally true and lets the
//! compare operation carry the guest's meaning. That is exact in both
//! directions: a guest asking for `Always` with writes off gets a pipeline
//! that passes every fragment and writes nothing, which is what a cleared
//! `depthTestEnable` would also have produced, and a guest asking for `Always`
//! *with* writes gets the write — which a cleared `depthTestEnable` would have
//! silently dropped.
//!
//! # One stencil enable for two faces
//!
//! Metal binds a stencil descriptor per face and either may be off. Vulkan has
//! a single `stencilTestEnable` covering both. The test is therefore engaged
//! when *either* face is, and the face that is not gets Metal's documented
//! `MTLStencilDescriptor` default — compare `Always`, every operation `Keep`,
//! full masks — which is a face that reads the buffer and changes nothing.
//!
//! That substitution is this layer's, not the protocol layer's: there, an
//! absent face is `None` and stays `None`, because the bytes behind a clear
//! enable bit are the guest's stale ring rather than a declaration.
//!
//! # The reference value is not here
//!
//! Metal sets the stencil reference on the encoder
//! (`setStencilReferenceValue:`), so it changes without the state changing. It
//! is `VK_DYNAMIC_STATE_STENCIL_REFERENCE` on this rail and is supplied per
//! draw; a plan that baked one in would key a pipeline cache on a value that
//! is not part of the state.
//!
//! # Planned, not created
//!
//! Nothing here creates a pipeline. The plan is a value, so every mapping is
//! tested with no GPU.

use ash::vk;
use reims_vgpu_core::depth_stencil::{DepthStencilState, StencilFace, StencilOperation as GuestOp};

/// `MTLCompareFunction` → `VkCompareOp`, spelled once.
///
/// The guest type is the sampler's — one comparison enumeration serves both a
/// sampler's compare-sample and this state's depth and stencil tests — so the
/// mapping is the sampler module's too, and this is a name for it rather than
/// a second copy. Two copies of a total mapping are two things that can come
/// to disagree, and nothing would fail if they did.
pub use crate::sampler::compare_op;

/// `MTLStencilOperation` → `VkStencilOp`.
///
/// The two enumerations name the same eight operations. They are written out
/// rather than cast, because a numeric coincidence between two vendors' enums
/// is not a contract and a reordering on either side would be silent.
#[must_use]
pub const fn stencil_op(guest: GuestOp) -> vk::StencilOp {
    match guest {
        GuestOp::Keep => vk::StencilOp::KEEP,
        GuestOp::Zero => vk::StencilOp::ZERO,
        GuestOp::Replace => vk::StencilOp::REPLACE,
        GuestOp::IncrementClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
        GuestOp::DecrementClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
        GuestOp::Invert => vk::StencilOp::INVERT,
        GuestOp::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
        GuestOp::DecrementWrap => vk::StencilOp::DECREMENT_AND_WRAP,
    }
}

/// One face, as `VkStencilOpState` would hold it.
///
/// Spelled out rather than held as a `vk::StencilOpState`, which is not `Eq` —
/// and a translation whose result cannot be compared is one whose mappings
/// cannot be asserted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FacePlan {
    pub fail_op: vk::StencilOp,
    pub pass_op: vk::StencilOp,
    pub depth_fail_op: vk::StencilOp,
    pub compare_op: vk::CompareOp,
    pub compare_mask: u32,
    pub write_mask: u32,
}

impl FacePlan {
    #[must_use]
    pub fn of(face: StencilFace) -> Self {
        Self {
            fail_op: stencil_op(face.stencil_failure),
            pass_op: stencil_op(face.depth_stencil_pass),
            depth_fail_op: stencil_op(face.depth_failure),
            compare_op: compare_op(face.compare),
            compare_mask: face.read_mask,
            write_mask: face.write_mask,
        }
    }

    /// The face substituted for one whose enable bit was clear. See the module
    /// doc.
    #[must_use]
    pub fn pass_through() -> Self {
        Self::of(StencilFace::DEFAULT)
    }

    /// The reference is deliberately zero here and set dynamically. See the
    /// module doc.
    pub const fn native(self) -> vk::StencilOpState {
        vk::StencilOpState {
            fail_op: self.fail_op,
            pass_op: self.pass_op,
            depth_fail_op: self.depth_fail_op,
            compare_op: self.compare_op,
            compare_mask: self.compare_mask,
            write_mask: self.write_mask,
            reference: 0,
        }
    }
}

/// A depth-stencil state as this rail would build it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DepthStencilPlan {
    /// Always true. See the module doc.
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_compare_op: vk::CompareOp,
    /// Always false: Metal has no depth-bounds test, so nothing the guest can
    /// declare turns this on.
    pub depth_bounds_test_enable: bool,
    pub stencil_test_enable: bool,
    pub front: FacePlan,
    pub back: FacePlan,
}

impl DepthStencilPlan {
    pub const fn native(self) -> vk::PipelineDepthStencilStateCreateInfo<'static> {
        vk::PipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
            depth_test_enable: as_bool32(self.depth_test_enable),
            depth_write_enable: as_bool32(self.depth_write_enable),
            depth_compare_op: self.depth_compare_op,
            depth_bounds_test_enable: as_bool32(self.depth_bounds_test_enable),
            stencil_test_enable: as_bool32(self.stencil_test_enable),
            front: self.front.native(),
            back: self.back.native(),
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
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

/// Translate a checked depth-stencil state.
///
/// Total: every state the guest API admits has a plan here, so there is no
/// refusal to return. The ordinals that could have failed were closed one
/// layer down.
#[must_use]
pub fn plan(state: &DepthStencilState) -> DepthStencilPlan {
    let stencil_test_enable = state.stencil_engaged();
    // A face is planned only where the test runs at all. With the test off,
    // both faces are the pass-through — which is what Vulkan ignores anyway,
    // and keeping the guest's ops there would make two states that behave
    // identically compare unequal in a pipeline cache key.
    let face = |present: Option<StencilFace>| {
        if stencil_test_enable {
            present.map_or_else(FacePlan::pass_through, FacePlan::of)
        } else {
            FacePlan::pass_through()
        }
    };
    DepthStencilPlan {
        depth_test_enable: true,
        depth_write_enable: state.depth_write(),
        depth_compare_op: compare_op(state.depth_compare()),
        depth_bounds_test_enable: false,
        stencil_test_enable,
        front: face(state.front()),
        back: face(state.back()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::depth_stencil::{
        DepthStencilShape, StencilFaceShape, MTL_STENCIL_OPERATION_INCREMENT_WRAP,
        MTL_STENCIL_OPERATION_REPLACE, MTL_STENCIL_OPERATION_ZERO,
    };
    use reims_vgpu_core::sampler::{
        CompareFunction, MTL_COMPARE_FUNCTION_ALWAYS, MTL_COMPARE_FUNCTION_GREATER,
        MTL_COMPARE_FUNCTION_LESS_EQUAL,
    };
    use std::collections::BTreeSet;

    fn face_shape() -> StencilFaceShape {
        StencilFaceShape {
            compare_function: MTL_COMPARE_FUNCTION_GREATER,
            stencil_failure_operation: MTL_STENCIL_OPERATION_ZERO,
            depth_failure_operation: MTL_STENCIL_OPERATION_INCREMENT_WRAP,
            depth_stencil_pass_operation: MTL_STENCIL_OPERATION_REPLACE,
            read_mask: 0xf0,
            write_mask: 0x0f,
        }
    }

    fn shape() -> DepthStencilShape {
        DepthStencilShape {
            depth_compare_function: MTL_COMPARE_FUNCTION_LESS_EQUAL,
            depth_write_enabled: true,
            front_stencil_enabled: false,
            back_stencil_enabled: false,
            front: StencilFaceShape::default(),
            back: StencilFaceShape::default(),
        }
    }

    fn planned(shape: DepthStencilShape) -> DepthStencilPlan {
        plan(&shape.checked().expect("a declaration the guest API admits"))
    }

    #[test]
    fn every_stencil_operation_maps_to_a_distinct_one() {
        let mapped: BTreeSet<i32> = GuestOp::ALL
            .iter()
            .map(|o| stencil_op(*o).as_raw())
            .collect();
        assert_eq!(mapped.len(), GuestOp::ALL.len());
        // The pair that differs only at the ends of the range, which is the
        // pair a cast would have got right by accident and a reordering wrong.
        assert_eq!(
            stencil_op(GuestOp::IncrementWrap),
            vk::StencilOp::INCREMENT_AND_WRAP
        );
        assert_eq!(
            stencil_op(GuestOp::IncrementClamp),
            vk::StencilOp::INCREMENT_AND_CLAMP
        );
        assert_eq!(stencil_op(GuestOp::Keep), vk::StencilOp::KEEP);
    }

    /// The claim the module doc makes about `depthTestEnable`: it is on for
    /// every state, so `Always` with writes on keeps the write that a cleared
    /// enable would have dropped.
    #[test]
    fn the_depth_test_is_always_enabled_so_a_write_is_never_dropped() {
        for compare in [MTL_COMPARE_FUNCTION_ALWAYS, MTL_COMPARE_FUNCTION_LESS_EQUAL] {
            for write in [false, true] {
                let plan = planned(DepthStencilShape {
                    depth_compare_function: compare,
                    depth_write_enabled: write,
                    ..shape()
                });
                assert!(plan.depth_test_enable);
                assert_eq!(plan.depth_write_enable, write);
                assert!(!plan.depth_bounds_test_enable);
            }
        }
        // The case that motivates it, named on its own.
        let always_writing = planned(DepthStencilShape {
            depth_compare_function: MTL_COMPARE_FUNCTION_ALWAYS,
            depth_write_enabled: true,
            ..shape()
        });
        assert!(always_writing.depth_test_enable);
        assert!(always_writing.depth_write_enable);
        assert_eq!(always_writing.depth_compare_op, vk::CompareOp::ALWAYS);
    }

    #[test]
    fn a_state_with_no_face_leaves_the_stencil_test_off() {
        let plan = planned(shape());
        assert!(!plan.stencil_test_enable);
        assert_eq!(plan.front, FacePlan::pass_through());
        assert_eq!(plan.back, FacePlan::pass_through());
    }

    /// One Vulkan enable for two Metal faces: the test runs when either face
    /// is bound, and the unbound one changes nothing.
    #[test]
    fn one_bound_face_engages_the_test_and_the_other_passes_through() {
        for (front_on, back_on) in [(true, false), (false, true), (true, true)] {
            let plan = planned(DepthStencilShape {
                front_stencil_enabled: front_on,
                back_stencil_enabled: back_on,
                front: face_shape(),
                back: face_shape(),
                ..shape()
            });
            assert!(plan.stencil_test_enable);

            let bound = FacePlan {
                fail_op: vk::StencilOp::ZERO,
                pass_op: vk::StencilOp::REPLACE,
                depth_fail_op: vk::StencilOp::INCREMENT_AND_WRAP,
                compare_op: vk::CompareOp::GREATER,
                compare_mask: 0xf0,
                write_mask: 0x0f,
            };
            assert_eq!(
                plan.front,
                if front_on {
                    bound
                } else {
                    FacePlan::pass_through()
                }
            );
            assert_eq!(
                plan.back,
                if back_on {
                    bound
                } else {
                    FacePlan::pass_through()
                }
            );
        }
    }

    /// The pass-through is a face that reads and changes nothing, which is
    /// what makes it a safe substitute for one the guest never declared.
    #[test]
    fn the_pass_through_face_writes_nothing_whatever_the_test_does() {
        let face = FacePlan::pass_through();
        assert_eq!(face.compare_op, vk::CompareOp::ALWAYS);
        for op in [face.fail_op, face.pass_op, face.depth_fail_op] {
            assert_eq!(op, vk::StencilOp::KEEP);
        }
    }

    /// Two states that behave identically must plan identically, or a pipeline
    /// cache keyed on the plan compiles the same pipeline twice.
    #[test]
    fn a_disengaged_test_plans_the_same_whatever_the_faces_held() {
        let bare = planned(shape());
        let with_bytes = planned(DepthStencilShape {
            front: face_shape(),
            back: face_shape(),
            ..shape()
        });
        assert_eq!(bare, with_bytes);
        assert!(!bare.stencil_test_enable);
    }

    #[test]
    fn every_comparison_reaches_the_depth_test_and_the_faces_alike() {
        let mut seen: BTreeSet<i32> = BTreeSet::new();
        for guest in CompareFunction::ALL {
            let plan = planned(DepthStencilShape {
                depth_compare_function: guest.ordinal(),
                front_stencil_enabled: true,
                front: StencilFaceShape {
                    compare_function: guest.ordinal(),
                    ..face_shape()
                },
                ..shape()
            });
            assert_eq!(plan.depth_compare_op, compare_op(guest));
            assert_eq!(plan.front.compare_op, compare_op(guest));
            assert!(seen.insert(plan.depth_compare_op.as_raw()));
        }
        assert_eq!(seen.len(), CompareFunction::ALL.len());
    }

    /// The reference is dynamic state, so nothing a plan holds carries one.
    #[test]
    fn the_native_face_leaves_the_reference_for_the_draw() {
        let plan = planned(DepthStencilShape {
            front_stencil_enabled: true,
            front: face_shape(),
            ..shape()
        });
        let native = plan.front.native();
        assert_eq!(native.reference, 0);
        assert_eq!(native.compare_mask, 0xf0);
        assert_eq!(native.write_mask, 0x0f);
    }

    #[test]
    fn the_native_state_carries_the_plan() {
        let plan = planned(DepthStencilShape {
            front_stencil_enabled: true,
            front: face_shape(),
            ..shape()
        });
        let native = plan.native();
        assert_eq!(native.depth_test_enable, vk::TRUE);
        assert_eq!(native.depth_write_enable, vk::TRUE);
        assert_eq!(native.depth_compare_op, vk::CompareOp::LESS_OR_EQUAL);
        assert_eq!(native.depth_bounds_test_enable, vk::FALSE);
        assert_eq!(native.stencil_test_enable, vk::TRUE);
        assert_eq!(native.front.compare_op, vk::CompareOp::GREATER);
        assert_eq!(native.back.compare_op, vk::CompareOp::ALWAYS);
        assert_eq!(
            native.s_type,
            vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO
        );
    }
}
