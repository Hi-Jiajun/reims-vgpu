//! What the Vulkan rail's resident registry can say about a present surface.
//!
//! Two questions, one file, because they are the same lookup asked for two
//! reasons: *would* a resident carry this present (a census/failure-channel
//! split at the drain), and *does* one, hand me its bytes (the capture). Both
//! resolve the surface through [`crate::backend::vulkan::present_identity`], and a
//! second spelling of that identity in either place would report a frame as
//! carried that the other then cannot find.
//!
//! Reached only through [`crate::backend::Backend`]; the drain and the capture
//! never name this rail.

use crate::backend::vulkan::engine;
use crate::backend::vulkan::present_identity::surface_identity;
use crate::model::DeviceState;

/// Would a resident carry the present this mapping names, at this geometry?
///
/// Asks [`engine::resident_presentable`], which shares `pools::slot_presentable`
/// with the window *publish* — the transaction that decides whether a resident
/// carries a present. Sharing the rule is the point rather than tidiness: a
/// looser predicate here would report a frame as carried that the publish then
/// refuses, which is a disagreement neither call site can see on its own — the
/// same shape as the publish/present split that once blanked the window.
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
/// The mapping's published frame from this rail's resident, as tight RGBA8.
///
/// `Backend::published_frame_rgba8` for this rail. `generation` is unused: the
/// engine's [`surface_identity`] already carries the evidence that decides which
/// image answers for a mapping at a geometry, and asking it to also match a
/// number this rail never wrote would decline every frame. The caller's
/// currency check still runs — it is what makes the question legitimate — it
/// simply is not the thing this rail keys on.
pub fn published_frame_rgba8(
    state: &DeviceState,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> Option<Vec<u8>> {
    use crate::protocol::pixel_format::RGBA8_BPP;
    let need = (width as usize)
        .checked_mul(RGBA8_BPP as usize)?
        .checked_mul(height as usize)?;
    let identity = surface_identity(state, mapping_id, width, height);
    let mut bgra = engine::read_resident_bgra(&identity, need).ok()?;
    if bgra.len() != need {
        return None;
    }
    // The engine reads BGRA8 for the console; every seed reader wants RGBA8.
    // In place — the same four bytes in a different order, and this readback is
    // already this function's own frame.
    crate::runtime::draw::swap_rb_channels_in_place(&mut bgra);
    Some(bgra)
}

/// [`CaptureRefusal`] for this rail's own readback miss.
///
/// The one place the rail's ladder is folded into the shared vocabulary, so a
/// reader of `reason=` and a reader of the probe's summary cannot be told two
/// different stories about one refusal.
fn refusal_for(miss: &engine::ResidentReadMiss) -> super::CaptureRefusal {
    use super::CaptureRefusal as Refusal;
    match miss {
        engine::ResidentReadMiss::UnknownIdentity { divergence } => match divergence {
            Some((engine::TargetKeyDivergence::Generation, _)) => Refusal::KeyGeneration,
            Some((engine::TargetKeyDivergence::Geometry, _)) => Refusal::KeyGeometry,
            Some((engine::TargetKeyDivergence::Other, _)) => Refusal::KeyOther,
            // `Namespace` is filtered out of the ladder (a key in another
            // namespace is not about this object), and `Absent` — like a rail
            // that could not read the ladder at all — is the plain missing
            // target.
            _ => Refusal::NoTarget,
        },
        engine::ResidentReadMiss::NoReadyContent { .. } => Refusal::ContentNotReady,
        engine::ResidentReadMiss::TexelNotScanout
        | engine::ResidentReadMiss::Readback
        | engine::ResidentReadMiss::Short { .. } => Refusal::ReadbackDeclined,
    }
}

pub fn try_capture_from_resident(
    state: &mut DeviceState,
    buf: &mut Vec<u8>,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> Result<(), super::CaptureRefusal> {
    let need = buf.len();
    let identity = surface_identity(state, mapping_id, width, height);
    let bgra = match engine::read_resident_bgra(&identity, need) {
        Ok(bgra) => bgra,
        Err(miss) => {
            let refusal = refusal_for(&miss);
            note_capture_miss(state, &identity, mapping_id, need, &miss, refusal);
            return Err(refusal);
        }
    };
    debug_assert_eq!(bgra.len(), need);
    // Move (not copy) the readback in; the untouched scratch returns to the pool.
    state.present.capture_scratch = std::mem::replace(buf, bgra);
    Ok(())
}

/// Account one resident-direct miss for
/// [`crate::config::CAPTURE_PROBE`], and nothing at all when it is off.
///
/// The step is the miss's own variant — not a second reading of the same
/// conditions — and the line carries the identity the capture asked under, the
/// registry's answer for it, and the two mapping ids the guest's screen is
/// caught between, because a reader of `present_capture FAIL` has none of the
/// three.
fn note_capture_miss(
    state: &DeviceState,
    identity: &engine::TargetIdentity,
    mapping_id: u32,
    need: usize,
    miss: &engine::ResidentReadMiss,
    refusal: super::CaptureRefusal,
) {
    use crate::runtime::scanout::capture_probe;
    if !capture_probe::enabled() {
        return;
    }
    // The identity carries the geometry it was built from, so the line reads it
    // there rather than taking a second copy of the same pair as an argument —
    // the same reason `surface_identity` is the only producer of both.
    let (width, height) = (identity.width(), identity.height());
    let what = match miss {
        engine::ResidentReadMiss::UnknownIdentity { divergence } => {
            let (how, held) = divergence.unwrap_or((engine::TargetKeyDivergence::Absent, None));
            format!(
                "how={how:?} held_generation={}",
                held.map(|g| g.to_string())
                    .unwrap_or_else(|| "none".to_owned())
            )
        }
        engine::ResidentReadMiss::NoReadyContent { width, height } => {
            format!("slot_extent={width}x{height} (registered, nothing vouched for it)")
        }
        engine::ResidentReadMiss::TexelNotScanout => {
            "the readback's texel cannot be handed over in scanout order".to_owned()
        }
        engine::ResidentReadMiss::Readback => "the readback itself declined".to_owned(),
        engine::ResidentReadMiss::Short { have } => {
            format!("readback_have={have} frame_need={need}")
        }
    };
    let signature = format!(
        "mid={mapping_id} {width}x{height} map_gen={} present_mapping={} frame_mapping={}",
        identity.generation(),
        state.present.present_mapping,
        state.present.frame_mapping,
    );
    capture_probe::note_step(refusal, &signature, || {
        format!(
            "{signature} {what} identity={identity:?} slot: {} \
             frame_valid={} frame_geom={}x{}",
            engine::capture_probe_report(identity),
            state.present.frame_valid as u8,
            state.present.frame_width,
            state.present.frame_height,
        )
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::scanout::CaptureRefusal as Refusal;

    /// The rail's ladder folds into the shared vocabulary in exactly one place,
    /// and each fold has to keep its meaning: a surface registered under
    /// another generation is a *key* fault an operator repairs differently from
    /// a surface nothing names at all.
    ///
    /// The `None` arm is the shipping one — the ladder is taken only while the
    /// probe is on — so it must read as the plain missing target rather than as
    /// a class of its own.
    #[test]
    fn the_ladder_folds_onto_the_shared_vocabulary() {
        use engine::ResidentReadMiss as Miss;
        use engine::TargetKeyDivergence as How;

        let unknown = |how| Miss::UnknownIdentity {
            divergence: Some((how, Some(3))),
        };
        assert_eq!(refusal_for(&unknown(How::Absent)), Refusal::NoTarget);
        assert_eq!(
            refusal_for(&unknown(How::Generation)),
            Refusal::KeyGeneration
        );
        assert_eq!(refusal_for(&unknown(How::Geometry)), Refusal::KeyGeometry);
        assert_eq!(refusal_for(&unknown(How::Other)), Refusal::KeyOther);
        assert_eq!(refusal_for(&unknown(How::Namespace)), Refusal::NoTarget);
        assert_eq!(
            refusal_for(&Miss::UnknownIdentity { divergence: None }),
            Refusal::NoTarget,
            "a rail that cannot read the ladder still names the missing target"
        );
        assert_eq!(
            refusal_for(&Miss::NoReadyContent {
                width: 64,
                height: 32
            }),
            Refusal::ContentNotReady
        );
        assert_eq!(
            refusal_for(&Miss::TexelNotScanout),
            Refusal::ReadbackDeclined
        );
        assert_eq!(refusal_for(&Miss::Readback), Refusal::ReadbackDeclined);
        assert_eq!(
            refusal_for(&Miss::Short { have: 1 }),
            Refusal::ReadbackDeclined
        );
    }
}
