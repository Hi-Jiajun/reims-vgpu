//! What a transaction touches, how precisely that is known, and what that
//! implies about ordering.
//!
//! # Precision is a ladder, and the rung is part of the answer
//!
//! An operation's access can be known at four different precisions, and the
//! model has to carry which one it got rather than flattening them:
//!
//! 1. the exact byte range or image subresource, when the command names it;
//! 2. the whole backing, when a resource table names participation but no
//!    range — or the whole heap, when a heap-use record declares indirect
//!    access to everything allocated from it without naming a resource;
//! 3. the submission domain alone, when participation is incomplete;
//! 4. a typed refusal, when the *operation* cannot be executed correctly.
//!
//! Rung 4 is the one that is easy to reach for and wrong. Lack of a concurrency
//! proof is not a reason to reject valid guest work: an operation whose access
//! is imprecisely known still executes, at rung 3, ordered by its domain. Only
//! an operation the device cannot perform is refused, and that decision belongs
//! to the closure ledger rather than to this compiler.
//!
//! # Coarse and fine keys have to meet
//!
//! The rungs are not separate namespaces. A draw that names a level of a
//! texture and a resource-lifecycle packet that names the whole backing are
//! talking about the same memory, and a conflict test that compared them by
//! variant would let them past each other. So conflict is decided on the
//! memory, not on the precision: two keys conflict when the memory they could
//! refer to overlaps, and a coarser key overlaps everything inside it.
//!
//! # Read against read is the only free pair
//!
//! Within one key, a proven read depends on the preceding writer; a proven
//! write depends on the preceding writer *and* every preceding reader; two
//! proven reads create no edge. An access whose mode is not established
//! conflicts with everything, visibly — [`AccessMode::Unknown`] is a distinct
//! variant precisely so a census can count how much ordering is being bought
//! with ignorance.

use crate::identity::ChannelId;

/// The canonical identity of a piece of backing memory.
///
/// Aliasing is decided from contract-declared backing relationships, and this
/// is that decision's result: two resources that share backing share a
/// `BackingId`. Resource names alone never prove or disprove aliasing, so
/// nothing here is derived from a name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BackingId(pub u64);

/// A heap, and the generation of its membership.
///
/// The generation is part of the identity because heap-use participation is
/// evaluated at the command point: a heap whose membership changed between two
/// commands is not the same participation domain for both, and comparing them
/// by heap number alone would silently order against the wrong set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HeapId {
    pub id: u64,
    pub membership_generation: u64,
}

/// A half-open byte range within one backing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteRange {
    pub offset: u64,
    pub length: u64,
}

impl ByteRange {
    /// Whether two ranges share a byte.
    ///
    /// A zero-length range overlaps nothing, including itself: it names no byte,
    /// so an operation carrying one touches nothing and cannot conflict.
    #[must_use]
    pub const fn overlaps(self, other: ByteRange) -> bool {
        if self.length == 0 || other.length == 0 {
            return false;
        }
        let self_end = self.offset.saturating_add(self.length);
        let other_end = other.offset.saturating_add(other.length);
        self.offset < other_end && other.offset < self_end
    }
}

/// A half-open window of an image's levels and slices.
///
/// Plane is exact rather than a range: a plane is a separate memory layout, not
/// a coordinate within one, so "planes 0..2" is two accesses and not one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubresourceRange {
    pub base_level: u32,
    pub level_count: u32,
    pub base_slice: u32,
    pub slice_count: u32,
    pub plane: u32,
}

impl SubresourceRange {
    #[must_use]
    pub const fn overlaps(self, other: SubresourceRange) -> bool {
        if self.plane != other.plane {
            return false;
        }
        span_overlaps(
            self.base_level,
            self.level_count,
            other.base_level,
            other.level_count,
        ) && span_overlaps(
            self.base_slice,
            self.slice_count,
            other.base_slice,
            other.slice_count,
        )
    }
}

const fn span_overlaps(a_base: u32, a_count: u32, b_base: u32, b_count: u32) -> bool {
    if a_count == 0 || b_count == 0 {
        return false;
    }
    let a_end = a_base.saturating_add(a_count);
    let b_end = b_base.saturating_add(b_count);
    a_base < b_end && b_base < a_end
}

/// A resource's backing, and the heap it was allocated from if it has one.
///
/// The heap travels with the key because heap-use participation names a heap
/// and not its members: without it, a heap declaration and a resource access
/// have nothing to compare and the coarser rung would silently order against
/// nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceKey {
    pub backing: BackingId,
    pub heap: Option<HeapId>,
}

/// How precisely an access is known.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessKey {
    /// The command named an exact byte range.
    Range(ResourceKey, ByteRange),
    /// The command named exact levels, slices and a plane.
    Subresource(ResourceKey, SubresourceRange),
    /// Participation is named but no range is: the whole backing.
    Whole(ResourceKey),
    /// A heap-use record: indirect access to everything allocated from this
    /// heap, with no per-resource usage named.
    Heap(HeapId),
    /// Participation is incomplete. Nothing about which memory is touched is
    /// established, so ordering comes from the submission domain alone.
    ///
    /// Not a refusal. An operation here still executes.
    DomainOnly,
}

impl AccessKey {
    /// Which rung of the precision ladder this key sits on, for the census that
    /// prices how much ordering is being bought with imprecision.
    #[must_use]
    pub const fn rung(self) -> u8 {
        match self {
            Self::Range(..) | Self::Subresource(..) => 1,
            Self::Whole(_) | Self::Heap(_) => 2,
            Self::DomainOnly => 3,
        }
    }

    const fn resource(self) -> Option<ResourceKey> {
        match self {
            Self::Range(r, _) | Self::Subresource(r, _) | Self::Whole(r) => Some(r),
            Self::Heap(_) | Self::DomainOnly => None,
        }
    }

    /// Whether the memory these two keys could refer to overlaps.
    ///
    /// Deliberately decided on the memory rather than on the variant: a draw
    /// naming one level of a texture and a lifecycle packet naming the whole
    /// backing are talking about the same bytes, and comparing them by shape
    /// would let them past each other.
    #[must_use]
    pub fn may_alias(self, other: AccessKey) -> bool {
        // Incomplete participation could be anything, so it meets everything.
        if matches!(self, Self::DomainOnly) || matches!(other, Self::DomainOnly) {
            return true;
        }
        match (self, other) {
            // Two heap declarations meet when they are the same heap at the
            // same membership; a different membership generation is a
            // different set of resources, so it is a different domain.
            (Self::Heap(a), Self::Heap(b)) => a == b,
            // A heap declaration meets every resource allocated from it.
            (Self::Heap(h), key) | (key, Self::Heap(h)) => {
                key.resource().is_some_and(|r| r.heap == Some(h))
            }
            (a, b) => {
                let (Some(ra), Some(rb)) = (a.resource(), b.resource()) else {
                    return false;
                };
                if ra.backing != rb.backing {
                    return false;
                }
                match (a, b) {
                    (Self::Range(_, x), Self::Range(_, y)) => x.overlaps(y),
                    (Self::Subresource(_, x), Self::Subresource(_, y)) => x.overlaps(y),
                    // A byte range and a subresource window are two coordinate
                    // systems over one backing, and nothing here relates them.
                    // The honest answer is that they may alias — narrowing it
                    // would need the image's layout, which is a decision the
                    // executor owns and this crate cannot see.
                    _ => true,
                }
            }
        }
    }
}

/// What an access does to the memory it names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
    /// The direction is not established.
    ///
    /// A separate variant rather than `ReadWrite`, even though it orders the
    /// same way: the census has to be able to say how many edges exist because
    /// something is genuinely read-modify-write and how many exist because
    /// nobody knows, and a conservative answer that cannot be counted is a
    /// conservative answer nobody will ever narrow.
    Unknown,
}

impl AccessMode {
    #[must_use]
    pub const fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite | Self::Unknown)
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::ReadWrite => "read_write",
            Self::Unknown => "unknown",
        }
    }
}

/// A resource's content version.
///
/// Content transfers are keyed to a version transition: no transfer happens
/// without one, and none repeats for the same key. An access declares the
/// version it consumes and, when it writes, the version it will produce — and
/// the produced version is *reserved* during planning and committed only after
/// the work completes, so a reader planned against it waits for the completion
/// rather than for the plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ContentVersion(pub u64);

impl ContentVersion {
    #[must_use]
    pub const fn next(self) -> ContentVersion {
        ContentVersion(self.0.wrapping_add(1))
    }
}

/// One access an operation declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccessIntent {
    /// The submission ordering domain the access belongs to. Carried on the
    /// access rather than looked up, because a conflict test that had to reach
    /// for it is a conflict test that can be called without it.
    pub domain: ChannelId,
    pub key: AccessKey,
    pub mode: AccessMode,
    /// The API stages the access is declared for, as the wire carries them.
    /// Translated into host stage masks by an executor and never here.
    pub api_stages: u32,
    /// The version this access consumes, when one is established.
    pub input_content_version: Option<ContentVersion>,
    /// The version this access will produce. Reserved at planning, committed at
    /// completion; `None` for a pure read.
    pub output_content_version: Option<ContentVersion>,
}

/// Whether an earlier access and a later one require an ordering edge.
///
/// Read against read is the only free pair, and only when both directions are
/// established: an [`AccessMode::Unknown`] on either side conflicts, because
/// what it does not know might be a write.
///
/// Cross-domain accesses produce no edge from conflict alone. That is not an
/// oversight and not an optimisation: the contract leaves separate submission
/// domains unordered, and manufacturing an edge here would repair an
/// application data race into a guarantee this API does not make. Cross-domain
/// visibility comes from explicit synchronisation, and a host-safety
/// serialisation an executor needs is a separate kind of edge that cannot order
/// guest-visible publication.
#[must_use]
pub fn requires_edge(earlier: &AccessIntent, later: &AccessIntent) -> bool {
    if earlier.domain != later.domain {
        return false;
    }
    if !earlier.mode.writes() && !later.mode.writes() {
        return false;
    }
    earlier.key.may_alias(later.key)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(backing: u64) -> ResourceKey {
        ResourceKey {
            backing: BackingId(backing),
            heap: None,
        }
    }

    fn intent(key: AccessKey, mode: AccessMode) -> AccessIntent {
        AccessIntent {
            domain: ChannelId(1),
            key,
            mode,
            api_stages: 0,
            input_content_version: None,
            output_content_version: None,
        }
    }

    #[test]
    fn a_zero_length_range_names_no_byte_and_so_conflicts_with_nothing() {
        let empty = ByteRange {
            offset: 0,
            length: 0,
        };
        assert!(!empty.overlaps(empty));
        assert!(!empty.overlaps(ByteRange {
            offset: 0,
            length: 16
        }));
    }

    #[test]
    fn adjacent_ranges_do_not_overlap_and_touching_ones_do() {
        let a = ByteRange {
            offset: 0,
            length: 16,
        };
        assert!(!a.overlaps(ByteRange {
            offset: 16,
            length: 16
        }));
        assert!(a.overlaps(ByteRange {
            offset: 15,
            length: 1
        }));
    }

    /// The failure this exists to prevent: a fine key and a coarse key over one
    /// backing compared by shape rather than by memory, so a resource delete
    /// naming the whole backing passes a draw naming one level of it.
    #[test]
    fn a_whole_backing_meets_every_precise_access_inside_it() {
        let whole = AccessKey::Whole(key(7));
        let sub = AccessKey::Subresource(
            key(7),
            SubresourceRange {
                base_level: 3,
                level_count: 1,
                base_slice: 0,
                slice_count: 1,
                plane: 0,
            },
        );
        let range = AccessKey::Range(
            key(7),
            ByteRange {
                offset: 4096,
                length: 64,
            },
        );
        assert!(whole.may_alias(sub));
        assert!(sub.may_alias(whole));
        assert!(whole.may_alias(range));
        // And a different backing still does not meet it.
        assert!(!whole.may_alias(AccessKey::Whole(key(8))));
    }

    /// Two coordinate systems over one backing, with nothing here relating
    /// them. The honest answer is that they may alias; narrowing it needs the
    /// image layout, which is the executor's to know.
    #[test]
    fn a_byte_range_and_a_subresource_over_one_backing_may_alias() {
        let range = AccessKey::Range(
            key(1),
            ByteRange {
                offset: 0,
                length: 4,
            },
        );
        let sub = AccessKey::Subresource(
            key(1),
            SubresourceRange {
                base_level: 9,
                level_count: 1,
                base_slice: 0,
                slice_count: 1,
                plane: 0,
            },
        );
        assert!(range.may_alias(sub));
    }

    #[test]
    fn different_planes_are_different_memory() {
        let plane = |p| {
            AccessKey::Subresource(
                key(2),
                SubresourceRange {
                    base_level: 0,
                    level_count: 1,
                    base_slice: 0,
                    slice_count: 1,
                    plane: p,
                },
            )
        };
        assert!(!plane(0).may_alias(plane(1)));
        assert!(plane(1).may_alias(plane(1)));
    }

    /// A heap declaration names no resource, so it can only meet one through
    /// membership — and membership at the command point, since a heap whose
    /// contents changed is a different participation domain.
    #[test]
    fn a_heap_declaration_meets_its_members_and_not_a_later_membership() {
        let heap = HeapId {
            id: 5,
            membership_generation: 2,
        };
        let member = AccessKey::Whole(ResourceKey {
            backing: BackingId(11),
            heap: Some(heap),
        });
        let stranger = AccessKey::Whole(key(11));
        assert!(AccessKey::Heap(heap).may_alias(member));
        assert!(member.may_alias(AccessKey::Heap(heap)));
        assert!(
            !AccessKey::Heap(heap).may_alias(stranger),
            "a resource with no heap is not in one"
        );
        let later = HeapId {
            id: 5,
            membership_generation: 3,
        };
        assert!(
            !AccessKey::Heap(heap).may_alias(AccessKey::Heap(later)),
            "the same heap at a different membership is a different set"
        );
    }

    /// Incomplete participation could be anything, so it meets everything —
    /// and it is still not a refusal.
    #[test]
    fn incomplete_participation_meets_everything() {
        for other in [
            AccessKey::Whole(key(1)),
            AccessKey::Heap(HeapId {
                id: 1,
                membership_generation: 0,
            }),
            AccessKey::DomainOnly,
        ] {
            assert!(AccessKey::DomainOnly.may_alias(other));
            assert!(other.may_alias(AccessKey::DomainOnly));
        }
        assert_eq!(AccessKey::DomainOnly.rung(), 3);
    }

    #[test]
    fn read_against_read_is_the_only_free_pair() {
        let k = AccessKey::Whole(key(3));
        let r = intent(k, AccessMode::Read);
        let w = intent(k, AccessMode::Write);
        let rw = intent(k, AccessMode::ReadWrite);
        let u = intent(k, AccessMode::Unknown);
        assert!(!requires_edge(&r, &r));
        assert!(requires_edge(&r, &w));
        assert!(requires_edge(&w, &r));
        assert!(requires_edge(&w, &w));
        assert!(requires_edge(&rw, &r));
        assert!(
            requires_edge(&r, &u) && requires_edge(&u, &r),
            "an unestablished direction might be a write, so it is not free"
        );
    }

    /// Conflict alone does not cross submission domains. Manufacturing that
    /// edge would repair an application data race into a guarantee this API
    /// does not make.
    #[test]
    fn a_conflict_across_domains_creates_no_edge() {
        let k = AccessKey::Whole(key(4));
        let mut a = intent(k, AccessMode::Write);
        let mut b = intent(k, AccessMode::Read);
        assert!(requires_edge(&a, &b), "same domain, write then read");
        a.domain = ChannelId(1);
        b.domain = ChannelId(2);
        assert!(!requires_edge(&a, &b));
    }

    #[test]
    fn the_rungs_are_ordered_from_exact_to_domain_only() {
        let k = key(1);
        assert_eq!(
            AccessKey::Range(
                k,
                ByteRange {
                    offset: 0,
                    length: 1
                }
            )
            .rung(),
            1
        );
        assert_eq!(AccessKey::Whole(k).rung(), 2);
        assert_eq!(
            AccessKey::Heap(HeapId {
                id: 0,
                membership_generation: 0
            })
            .rung(),
            2
        );
        assert_eq!(AccessKey::DomainOnly.rung(), 3);
    }
}
