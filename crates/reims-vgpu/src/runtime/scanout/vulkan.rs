//! What the Vulkan rail's resident registry can say about a present surface.
//!
//! Two questions, one file, because they are the same lookup asked for two
//! reasons: *would* a resident carry this present (a census/failure-channel
//! split at the drain), and *does* one, hand me its bytes (the capture). Both
//! resolve the surface through [`crate::runtime::present_identity`], and a
//! second spelling of that identity in either place would report a frame as
//! carried that the other then cannot find.
//!
//! Reached only through [`crate::backend::Backend`]; the drain and the capture
//! never name this rail.

use crate::backend::vulkan::engine;
use crate::model::DeviceState;
use crate::runtime::present_identity::surface_identity;

/// Would a resident carry the present this mapping names, at this geometry?
///
/// Asks [`engine::resident_presentable`], which shares `pools::slot_presentable`
/// with the window presenter's own selection. Sharing the rule is the point
/// rather than tidiness: a looser predicate here would report a frame as carried
/// that the presenter then refuses, which is a disagreement neither call site
/// can see on its own — the same shape as the publish/present split that once
/// blanked the window.
pub fn present_resident_carries(
    state: &DeviceState,
    mapping: u32,
    width: u32,
    height: u32,
) -> Option<bool> {
    let identity = surface_identity(state, mapping, width, height);
    Some(engine::resident_presentable(&identity, width, height))
}

/// Fill `buf` from the mapping's GPU resident, without any guest-page scatter.
///
/// On `true` `buf` holds tight BGRA8; on `false` `buf` is untouched. A miss is
/// an expected steady-state condition (cold mid / no resident yet), so it is
/// counted in the `capture_source` census rather than logged per present.
pub fn try_capture_from_resident(
    state: &mut DeviceState,
    buf: &mut Vec<u8>,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> bool {
    let need = buf.len();
    let identity = surface_identity(state, mapping_id, width, height);
    let Some(bgra) = engine::read_resident_bgra(&identity, need) else {
        return false;
    };
    debug_assert_eq!(bgra.len(), need);
    // Move (not copy) the readback in; the untouched scratch returns to the pool.
    state.present.capture_scratch = std::mem::replace(buf, bgra);
    true
}
