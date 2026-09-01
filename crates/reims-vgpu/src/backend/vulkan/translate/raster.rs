//! Cull mode, winding, fill mode, depth clip mode, compare function, stencil
//! operation and index type → their Vulkan spellings.
//!
//! Metal and Vulkan agree on the *ordering* of several of these enums, which
//! makes a numeric cast tempting and wrong: the agreement is a coincidence of
//! two independent specs, not a contract, and a cast turns a future divergence
//! into silently wrong rasterization. Every arm is spelled out so the compiler
//! catches a drift instead of the screen.

use ash::vk;

use super::reason::TranslateReason;
use crate::backend::vulkan::engine::{
    IndexType, SamplerCompareFunction, StencilOp, VisibilityResultMode,
};

/// `MTLCompareFunction` (SDK numeric values). Depth test, stencil test and
/// sampler compare all carry this same Metal enum.
pub fn compare_function(mtl: u32) -> Result<SamplerCompareFunction, TranslateReason> {
    Ok(match mtl {
        0 => SamplerCompareFunction::Never,
        1 => SamplerCompareFunction::Less,
        2 => SamplerCompareFunction::Equal,
        3 => SamplerCompareFunction::LessEqual,
        4 => SamplerCompareFunction::Greater,
        5 => SamplerCompareFunction::NotEqual,
        6 => SamplerCompareFunction::GreaterEqual,
        7 => SamplerCompareFunction::Always,
        other => return Err(TranslateReason::UnknownCompareFunction(other)),
    })
}

/// `MTLStencilOperation` (SDK numeric values).
pub fn stencil_operation(mtl: u32) -> Result<StencilOp, TranslateReason> {
    Ok(match mtl {
        0 => StencilOp::Keep,
        1 => StencilOp::Zero,
        2 => StencilOp::Replace,
        3 => StencilOp::IncrementClamp,
        4 => StencilOp::DecrementClamp,
        5 => StencilOp::Invert,
        6 => StencilOp::IncrementWrap,
        7 => StencilOp::DecrementWrap,
        other => return Err(TranslateReason::UnknownStencilOperation(other)),
    })
}

/// `MTLIndexType` (SDK numeric values).
///
/// The shared runtime loader owns the typed refusal because both Metal and
/// Vulkan consume it; `None` therefore remains a classification here, and the
/// caller turns it into `IndexLoadReason::TypeUnsupported`.
pub fn index_type(mtl: u32) -> Option<IndexType> {
    match mtl {
        0 => Some(IndexType::U16),
        1 => Some(IndexType::U32),
        _ => None,
    }
}

pub fn vk_index_type(index: IndexType) -> vk::IndexType {
    match index {
        IndexType::U16 => vk::IndexType::UINT16,
        IndexType::U32 => vk::IndexType::UINT32,
    }
}

/// `MTLVisibilityResultMode` (SDK numeric values) → whether a draw arms an
/// occlusion query, and what it counts.
///
/// `Ok(None)` is `MTLVisibilityResultModeDisabled`, the Metal default: the guest
/// disarmed the query, so the draw runs without one. That is why this returns
/// an `Option` inside the `Result` rather than folding `0` into the error arm —
/// disarming is a thing the guest is entitled to ask for, and an unknown
/// ordinal is not.
pub fn visibility_result_mode(mtl: u32) -> Result<Option<VisibilityResultMode>, TranslateReason> {
    use crate::protocol::visibility::VISIBILITY_RESULT_MODE_DISABLED;
    Ok(match mtl {
        // The one arm that is a *meaning* rather than a mode, so it is the one
        // arm spelled from the contract rather than as a literal beside its
        // neighbours.
        VISIBILITY_RESULT_MODE_DISABLED => None,
        1 => Some(VisibilityResultMode::Boolean),
        2 => Some(VisibilityResultMode::Counting),
        other => return Err(TranslateReason::UnknownVisibilityResultMode(other)),
    })
}

/// The query-control flags an armed mode records with.
///
/// `PRECISE` is what makes the result an exact sample count rather than a
/// non-zero-if-any. It requires `VkPhysicalDeviceFeatures::occlusionQueryPrecise`;
/// the caller gates on it, because the alternative — quietly recording without
/// the bit — hands a counting guest a number that is neither the count nor
/// recognisably wrong, which is the same failure `vk_polygon_mode` above
/// refuses to make for wireframe.
pub fn vk_query_control_flags(mode: VisibilityResultMode) -> vk::QueryControlFlags {
    match mode {
        VisibilityResultMode::Boolean => vk::QueryControlFlags::empty(),
        VisibilityResultMode::Counting => vk::QueryControlFlags::PRECISE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every primitive type this device *advertises* has an arm here.
    ///
    /// [`crate::protocol::draw::EXECUTABLE_PRIMITIVE_TYPES`] is what the guest
    /// reads as permission, so a bit set there without an arm here is a draw the
    /// guest was invited to make and this rail refuses. Both directions are
    /// asserted: an arm without the bit would be a type this device can execute
    /// and has told the guest not to use.
    #[test]
    fn the_advertised_primitive_types_are_the_executable_ones() {
        for mtl in 0..=8u32 {
            assert_eq!(
                reims_vgpu_core::topology::PrimitiveType::parse(mtl).is_some(),
                crate::protocol::draw::primitive_type_executable(mtl),
                "primitive type {mtl}: advertisement and translation disagree"
            );
        }
    }

    /// Every SDK value in range maps, and the first one past the end declines
    /// by its own slug rather than a shared one.
    #[test]
    fn each_raster_enum_is_total_over_its_sdk_range() {
        for mtl in 0..=7u32 {
            assert!(compare_function(mtl).is_ok(), "compare {mtl}");
            assert!(stencil_operation(mtl).is_ok(), "stencil {mtl}");
        }
        assert_eq!(
            compare_function(8).unwrap_err(),
            TranslateReason::UnknownCompareFunction(8)
        );
        assert_eq!(
            stencil_operation(8).unwrap_err(),
            TranslateReason::UnknownStencilOperation(8)
        );
    }

    /// The exact wire order of `MTLCompareFunction`, value by value.
    ///
    /// Injectivity below proves no two values collide; it does not prove the
    /// table is not *rotated*. A rotation still round-trips and still renders —
    /// it just inverts occlusion for every 3D draw — so the mapping is pinned
    /// arm by arm. (Moved here from `runtime/draw/mod.rs`, which held a second
    /// copy of this table; the assertion outlived the duplicate.)
    #[test]
    fn compare_function_matches_the_metal_abi_order() {
        use SamplerCompareFunction as C;
        assert_eq!(compare_function(0), Ok(C::Never));
        assert_eq!(compare_function(1), Ok(C::Less));
        assert_eq!(compare_function(2), Ok(C::Equal));
        assert_eq!(compare_function(3), Ok(C::LessEqual));
        assert_eq!(compare_function(4), Ok(C::Greater));
        assert_eq!(compare_function(5), Ok(C::NotEqual));
        assert_eq!(compare_function(6), Ok(C::GreaterEqual));
        assert_eq!(compare_function(7), Ok(C::Always));
        assert_eq!(
            compare_function(99).unwrap_err(),
            TranslateReason::UnknownCompareFunction(99)
        );
    }

    /// Same, for `MTLStencilOperation`. The increment/decrement pairs are the
    /// transcription hazard: clamp and wrap differ only in overflow behaviour.
    #[test]
    fn stencil_operation_matches_the_metal_abi_order() {
        use StencilOp as O;
        assert_eq!(stencil_operation(0), Ok(O::Keep));
        assert_eq!(stencil_operation(1), Ok(O::Zero));
        assert_eq!(stencil_operation(2), Ok(O::Replace));
        assert_eq!(stencil_operation(3), Ok(O::IncrementClamp));
        assert_eq!(stencil_operation(4), Ok(O::DecrementClamp));
        assert_eq!(stencil_operation(5), Ok(O::Invert));
        assert_eq!(stencil_operation(6), Ok(O::IncrementWrap));
        assert_eq!(stencil_operation(7), Ok(O::DecrementWrap));
        assert_eq!(
            stencil_operation(99).unwrap_err(),
            TranslateReason::UnknownStencilOperation(99)
        );
    }

    /// One Apple enum, spelled once here and once in
    /// `backend::metal::mtl_enum::visibility_result_mode`, with nothing in the
    /// toolchain comparing them. A mode this arm records and that one refuses
    /// is a guest that culls correctly on one host and reads a stale word on
    /// the other, so both are held to [`crate::protocol::visibility`] — this
    /// one as a test, the Metal one as a `const` block, because its tests run
    /// on no machine anybody edits from.
    #[test]
    fn the_recorded_visibility_modes_are_the_ones_the_contract_names() {
        use crate::protocol::visibility::{
            visibility_result_mode_recordable, VISIBILITY_RESULT_MODE_DISABLED,
            VISIBILITY_RESULT_MODE_SWEEP_END,
        };
        for mtl in 0..VISIBILITY_RESULT_MODE_SWEEP_END {
            let recorded = matches!(visibility_result_mode(mtl), Ok(Some(_)));
            assert_eq!(
                recorded,
                visibility_result_mode_recordable(mtl),
                "ordinal {mtl}: this arm and the device contract disagree about \
                 whether an occlusion query armed with it is recorded"
            );
        }
        // The disarming ordinal is `Ok(None)` rather than a refusal: it is the
        // absence of a query, which is a thing a stream legitimately says.
        assert_eq!(
            visibility_result_mode(VISIBILITY_RESULT_MODE_DISABLED),
            Ok(None)
        );
    }

    #[test]
    fn index_types_map_by_width() {
        assert_eq!(index_type(0), Some(IndexType::U16));
        assert_eq!(index_type(1), Some(IndexType::U32));
        assert_eq!(index_type(2), None);
        assert_eq!(vk_index_type(IndexType::U16), vk::IndexType::UINT16);
        assert_eq!(vk_index_type(IndexType::U32), vk::IndexType::UINT32);
    }
}
