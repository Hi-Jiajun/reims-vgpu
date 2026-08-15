//! The `MTLLoadAction` and `MTLStoreAction` ordinals a render-pass attachment
//! prefix carries, and the closed sets this device implements.
//!
//! Two adjacent 16-bit words of every colour, depth and stencil attachment. The
//! guest writes the Metal SDK's own ordinals into them, so the numbers here are
//! simultaneously a wire fact and an SDK fact — which is why the conversion on
//! the Metal encode path is an identity: nothing is mapped, only widened.
//!
//! # Why they live here rather than in the decoder or the backend
//!
//! They were declared twice. `runtime::decode::render` had them as `u16` under
//! `PASS_*`, `backend::metal::abi` as `u32` under `REIMS_VGPU_MTL_*`, five
//! ordinals each, and nothing in the toolchain compared the two — the only
//! thing that touched both was an identity `match` in [`crate::runtime::draw`]
//! that read as a translation table. A value that arrives on the wire and is
//! consumed by both backends belongs beside the other wire/SDK numbers, per this
//! module tree's own doc; `backend::metal::abi` keeps its spelling because that
//! file is a mirror of an archived C header and the mirror is its provenance,
//! with `const` assertions there pinning the two equal on every arm that
//! compiles it.
//!
//! # The widths differ, and that is the whole conversion
//!
//! The attachment prefix spells an action in 16 bits; the Metal C shim takes an
//! `MTLLoadAction`/`MTLStoreAction` as `uint32_t`. So the declaration that
//! crosses to C is `u32` and this one is `u16`, and everything between them is a
//! widening — see [`crate::runtime::draw`]'s `map_load_action`, which returns
//! DontCare for a value outside the set and widens every value inside it.

/// `MTLLoadActionDontCare` — the attachment's prior contents may be discarded.
pub const MTL_LOAD_ACTION_DONT_CARE: u16 = 0;
/// `MTLLoadActionLoad` — the pass composites onto the attachment's contents.
pub const MTL_LOAD_ACTION_LOAD: u16 = 1;
/// `MTLLoadActionClear` — the attachment starts at the record's clear value.
pub const MTL_LOAD_ACTION_CLEAR: u16 = 2;

/// `MTLStoreActionDontCare` — the pass's result for this attachment is dropped.
pub const MTL_STORE_ACTION_DONT_CARE: u16 = 0;
/// `MTLStoreActionStore` — the pass's result is written back to the attachment.
pub const MTL_STORE_ACTION_STORE: u16 = 1;
/// `MTLStoreActionMultisampleResolve` — resolve into the attachment's
/// single-sample destination and discard the multisample source.
pub const MTL_STORE_ACTION_MULTISAMPLE_RESOLVE: u16 = 2;
/// `MTLStoreActionStoreAndMultisampleResolve` — preserve both images.
pub const MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE: u16 = 3;

/// Whether `raw` is one of the three `MTLLoadAction` values.
///
/// The set is *closed* in the same sense as [`crate::contract::dispatch`]'s:
/// `MTLLoadAction` has exactly these three, so a fourth ordinal is a corrupt
/// record or a wrong wire offset rather than a guest feature this device has no
/// contract for yet.
#[must_use]
pub fn is_declared_load_action(raw: u16) -> bool {
    raw <= MTL_LOAD_ACTION_CLEAR
}

/// Whether `raw` is one of the four store actions this device decodes by name.
///
/// The remaining SDK values require additional state not represented by this
/// wire form. Backend capability is a separate question: recognizing an action
/// here does not authorize a backend to approximate it.
#[must_use]
pub fn is_declared_store_action(raw: u16) -> bool {
    raw <= MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE
}

/// Whether the action publishes a single-sample destination the guest may
/// subsequently read.
#[must_use]
pub fn store_action_publishes_single_sample(raw: u16) -> bool {
    matches!(
        raw,
        MTL_STORE_ACTION_STORE
            | MTL_STORE_ACTION_MULTISAMPLE_RESOLVE
            | MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The accepted load set is exactly the three declared ordinals.
    ///
    /// Swept past the top constant in both directions, because the predicate's
    /// job is to be closed: every value it rejects is substituted with DontCare
    /// by its callers, so an accidentally-accepted fourth ordinal would reach a
    /// Metal enum conversion that has no arm for it.
    #[test]
    fn only_the_three_declared_load_actions_are_accepted() {
        assert!(is_declared_load_action(MTL_LOAD_ACTION_DONT_CARE));
        assert!(is_declared_load_action(MTL_LOAD_ACTION_LOAD));
        assert!(is_declared_load_action(MTL_LOAD_ACTION_CLEAR));
        for raw in (MTL_LOAD_ACTION_CLEAR + 1)..=64u16 {
            assert!(
                !is_declared_load_action(raw),
                "{raw} is not a declared MTLLoadAction"
            );
        }
        assert!(!is_declared_load_action(u16::MAX));
    }

    /// The recognized store set is exactly the four actions represented here.
    #[test]
    fn only_the_four_named_store_actions_are_accepted() {
        assert!(is_declared_store_action(MTL_STORE_ACTION_DONT_CARE));
        assert!(is_declared_store_action(MTL_STORE_ACTION_STORE));
        assert!(is_declared_store_action(MTL_STORE_ACTION_MULTISAMPLE_RESOLVE));
        assert!(is_declared_store_action(
            MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE
        ));
        for raw in (MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE + 1)..=64u16 {
            assert!(
                !is_declared_store_action(raw),
                "{raw} is not a named MTLStoreAction"
            );
        }
        assert!(!is_declared_store_action(u16::MAX));
    }

    #[test]
    fn only_actions_with_a_single_sample_result_publish_one() {
        assert!(!store_action_publishes_single_sample(
            MTL_STORE_ACTION_DONT_CARE
        ));
        for action in [
            MTL_STORE_ACTION_STORE,
            MTL_STORE_ACTION_MULTISAMPLE_RESOLVE,
            MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE,
        ] {
            assert!(store_action_publishes_single_sample(action));
        }
        assert!(!store_action_publishes_single_sample(u16::MAX));
    }

    /// Neither predicate can see the two adjacent words swapped.
    ///
    /// The load set is a strict subset of the store set — `{0, 1, 2}` inside
    /// `{0, 1, 2, 3}` — and the attachment prefix carries the two in adjacent
    /// words. So a decode that swaps the words can still produce values both
    /// predicates accept. Pinned
    /// because it bounds what these two can be asked to prove: they narrow a
    /// value to its own contract, and no arrangement of them detects a field
    /// offset that is two bytes out.
    #[test]
    fn the_load_set_is_a_strict_subset_of_the_store_set() {
        let load: Vec<u16> = (0..=u16::MAX)
            .filter(|&r| is_declared_load_action(r))
            .collect();
        let store: Vec<u16> = (0..=u16::MAX)
            .filter(|&r| is_declared_store_action(r))
            .collect();
        assert!(
            load.iter().all(|r| store.contains(r)),
            "a load ordinal outside the store set would make a swap detectable"
        );
        assert_eq!(
            store.iter()
                .filter(|r| !load.contains(r))
                .collect::<Vec<_>>(),
            vec![&MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE]
        );
    }
}
