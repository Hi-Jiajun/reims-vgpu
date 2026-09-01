//! The content-representation operations, and the one of them that has teeth.
//!
//! # Why these survive into the model at all
//!
//! All eight are proven no-ops in the ledger, and the cells say why: "guest
//! pages are written directly; there is no host-private texture layout", "guest
//! pages are the single copy of resource content". Both sentences describe the
//! *placement* the current executor chose. Neither is a property of the guest's
//! request.
//!
//! The plan gives placement to the executor — it is the only layer allowed to
//! turn host capabilities into placement and transfer policy — and
//! [`crate::content`] already models a device-owned replica precisely because
//! an executor may choose one. The moment it does, `synchronizeResource:` stops
//! being free: it becomes a readback the guest's next CPU read depends on. A
//! model that had dropped the record at decode would have nothing to schedule
//! that readback from.
//!
//! So these are operations, and the same reasoning `crate::sync` applies to
//! barriers applies here: a host fact belongs to the host.
//!
//! # Only one of the four has a definable effect
//!
//! [`ContentDirective::Synchronize`] means the guest's own pages must be
//! current for the named content. That is a statement about
//! [`crate::content::Replica`] and it is checkable, so
//! [`ResourceStateOp::publication_requirement`] returns it.
//!
//! The other three do not get invented effects. `optimizeContentsFor…` is a
//! preference about a representation the model does not describe;
//! `invalidateCompressedTexture:` says a compression metadata this model has no
//! concept of is stale. Mapping either onto a replica discard would be a guess
//! with a cost: discarding a device replica that holds the *authority* destroys
//! content the guest never wrote. So they are carried, reported, and left for
//! an executor that has a representation to act on.

use crate::access::SubresourceRange;
use crate::content::Replica;
use crate::identity::ResourceId;
pub use reims_vgpu_protocol::resource_state::{ContentDirective, ContentGranularity};

/// A texture subresource named by a `slice:level:` record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SliceLevel {
    pub slice: u16,
    pub level: u16,
}

impl SliceLevel {
    /// The one level and one slice this names.
    #[must_use]
    pub const fn subresource(self) -> SubresourceRange {
        SubresourceRange {
            base_level: self.level as u32,
            level_count: 1,
            base_slice: self.slice as u32,
            slice_count: 1,
            plane: 0,
        }
    }
}

/// One content-representation operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceStateOp {
    pub directive: ContentDirective,
    pub resource: ResourceId,
    /// The subresource the record named, or `None` for the whole-resource form.
    ///
    /// `None` is "the record carried a ref alone", not "slice 0, level 0". A
    /// whole-texture synchronise covers every level, and collapsing it to the
    /// top one would publish a fraction of what the guest asked for.
    pub subresource: Option<SliceLevel>,
}

impl ResourceStateOp {
    /// The granularity the record was sent at.
    #[must_use]
    pub const fn granularity(&self) -> ContentGranularity {
        match self.subresource {
            Some(_) => ContentGranularity::SliceLevel,
            None => ContentGranularity::WholeResource,
        }
    }

    /// The replica that must be current when this operation completes.
    ///
    /// `Some(GuestPages)` for a synchronise and nothing for the rest. This is
    /// the model's whole semantic contribution here, and it is deliberately
    /// small: three of the four directives have no effect this layer can state
    /// without inventing one.
    #[must_use]
    pub const fn publication_requirement(&self) -> Option<Replica> {
        match self.directive {
            ContentDirective::Synchronize => Some(Replica::GuestPages),
            ContentDirective::OptimizeForCpu
            | ContentDirective::OptimizeForGpu
            | ContentDirective::InvalidateCompressed => None,
        }
    }

    /// Whether an executor is free to do nothing at all for this operation
    /// *once it has stated where the content is*.
    ///
    /// Never answered here — the method returns what the model knows, which is
    /// that a synchronise depends on placement and the other three depend on a
    /// representation. It exists so a caller asking "can I skip this" has to
    /// pass through a name that says the answer is not the model's.
    #[must_use]
    pub const fn depends_on_placement(&self) -> bool {
        matches!(self.directive, ContentDirective::Synchronize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ObjectListRef, SlotGeneration};
    use crate::operation::{classify, OperationClass, OperationHome};
    use reims_vgpu_protocol::closure::{Rail, LEDGER};
    use reims_vgpu_protocol::resource_state::content_request;

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn op(directive: ContentDirective, subresource: Option<SliceLevel>) -> ResourceStateOp {
        ResourceStateOp {
            directive,
            resource: res(1),
            subresource,
        }
    }

    /// Every content request the protocol names is classified as a
    /// resource-state operation, and every classified resource-state operation
    /// that carries an opcode is a content request.
    ///
    /// The second direction is the one that would fail first: the residency
    /// records also classify as `ResourceState`, and they are all unresolved —
    /// so if one of them is ever closed, this test says the payload vocabulary
    /// owes it a shape.
    #[test]
    fn the_content_requests_are_exactly_the_judged_resource_state_operations() {
        for op in LEDGER {
            let Some(opcode) = op.opcode else { continue };
            let classified = classify(op)
                == Some(OperationHome::Stream(OperationClass::ResourceState))
                && op.rail == Rail::Blit;
            assert_eq!(
                content_request(opcode).is_some() && op.rail == Rail::Blit,
                classified,
                "{:?} {opcode:#x} disagrees about being a content request",
                op.rail
            );
        }
    }

    /// The residency records are the resource-state operations that are *not*
    /// content requests, and every one of them is unresolved. That is what
    /// keeps them out of the vocabulary rather than a special case.
    #[test]
    fn the_residency_records_are_unresolved_and_so_have_no_payload() {
        let mut residency = 0;
        for op in LEDGER {
            let Some(opcode) = op.opcode else { continue };
            if op.rail == Rail::Blit || content_request(opcode).is_some() {
                continue;
            }
            if !op.selector.starts_with("use") {
                continue;
            }
            residency += 1;
            assert!(
                op.closure.blocks_cutover(),
                "{:?} {opcode:#x} is a closed residency row and now needs a payload",
                op.rail
            );
            assert_eq!(classify(op), None);
        }
        assert!(residency >= 5, "the useResource/useHeap family");
    }

    /// A whole-resource record is not slice 0 level 0.
    #[test]
    fn a_whole_resource_request_is_not_the_top_subresource() {
        let whole = op(ContentDirective::Synchronize, None);
        let top = op(
            ContentDirective::Synchronize,
            Some(SliceLevel { slice: 0, level: 0 }),
        );
        assert_ne!(whole, top);
        assert_eq!(whole.granularity(), ContentGranularity::WholeResource);
        assert_eq!(top.granularity(), ContentGranularity::SliceLevel);
    }

    /// Synchronise publishes to the guest's pages; nothing else claims an
    /// effect.
    #[test]
    fn only_synchronise_states_a_publication_requirement() {
        assert_eq!(
            op(ContentDirective::Synchronize, None).publication_requirement(),
            Some(Replica::GuestPages)
        );
        for directive in [
            ContentDirective::OptimizeForCpu,
            ContentDirective::OptimizeForGpu,
            ContentDirective::InvalidateCompressed,
        ] {
            assert_eq!(op(directive, None).publication_requirement(), None);
            assert!(!op(directive, None).depends_on_placement());
        }
        assert!(op(ContentDirective::Synchronize, None).depends_on_placement());
    }

    /// A slice/level record names exactly one subresource, so two synchronises
    /// of different levels are independent.
    #[test]
    fn two_levels_of_one_texture_are_separate_subresources() {
        let a = SliceLevel { slice: 0, level: 0 }.subresource();
        let b = SliceLevel { slice: 0, level: 1 }.subresource();
        assert!(!a.overlaps(b));
    }
}
