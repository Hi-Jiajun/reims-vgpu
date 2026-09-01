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

use crate::identity::{ChannelId, ResourceId};

/// The canonical identity of a piece of backing memory.
///
/// Aliasing is decided from contract-declared backing relationships, and this
/// is that decision's result: two resources that share backing share a
/// `BackingId`. Resource names alone never prove or disprove aliasing, so
/// nothing here is derived from a name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BackingId(pub u64);

/// A heap, and the generation of its membership at the point the record was
/// decoded.
///
/// The generation is carried because it is a decoded fact worth reporting —
/// it says *which* membership set the declaration was written against — but it
/// is deliberately not part of the aliasing question. A resource leaves a heap
/// only by being destroyed, so a declaration recorded at generation *N* and an
/// access recorded at generation *N+1* can still name the very same bytes;
/// requiring the generations to match would drop that edge, and a dropped
/// hazard edge is a race. [`HeapId::same_heap`] is therefore what the conflict
/// test asks, and the cost of asking it is over-ordering against resources
/// placed after the declaration — the sound direction, and the same
/// conservatism the heap rung already carries by naming no usage at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HeapId {
    pub id: u64,
    pub membership_generation: u64,
}

impl HeapId {
    /// Whether these name the same heap, whatever either one's membership was
    /// when it was recorded.
    #[must_use]
    pub const fn same_heap(self, other: HeapId) -> bool {
        self.id == other.id
    }
}

/// A half-open byte range within one backing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceKey {
    pub backing: BackingId,
    pub heap: Option<HeapId>,
}

/// How precisely an access is known.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
            // Two declarations of one heap meet, whichever membership each
            // was recorded against.
            (Self::Heap(a), Self::Heap(b)) => a.same_heap(b),
            // A heap declaration meets every resource allocated from it.
            (Self::Heap(h), key) | (key, Self::Heap(h)) => key
                .resource()
                .is_some_and(|r| r.heap.is_some_and(|rh| rh.same_heap(h))),
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

/// What an operation says it touches, before the resource is resolved.
///
/// An operation record names a *ref* and a region; it does not name a backing,
/// a heap, or a content version. Those come from resolution, which needs the
/// resource registry — a thing this module cannot and should not see. So the
/// operation's own claim is this, and [`Participation::resolve`] is the single
/// step that turns it into an [`AccessIntent`].
///
/// The split matters beyond tidiness: it is what makes "an operation declares
/// its exact participation" checkable at the operation, where the record's
/// fields are, rather than after a registry lookup has already had a chance to
/// widen it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Participation {
    pub resource: ResourceId,
    pub extent: ParticipationExtent,
    pub mode: AccessMode,
    /// The API stages the record declares, as the wire carries them.
    ///
    /// Zero for a transfer: a copy record names no shader stage, and the
    /// transfer stage a host needs is a translation an executor performs. A
    /// non-zero value here always came from a record that carried one.
    pub api_stages: u32,
}

/// How much of a resource an operation named.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticipationExtent {
    /// An exact byte range.
    Range(ByteRange),
    /// Exact levels, slices and a plane.
    Subresource(SubresourceRange),
    /// The record named the resource and nothing narrower.
    ///
    /// Not the same as unknown participation: the resource *is* named, so this
    /// still conflicts only with that resource's memory. It is the honest
    /// answer for a record like `generateMipmapsForTexture:`, whose extent is
    /// the texture's whole pyramid and whose level count the record does not
    /// carry.
    Whole,
}

impl Participation {
    /// Attach the resolved backing, submission domain and content versions.
    ///
    /// The versions are the caller's because they are the content authority's:
    /// this type knows the operation reads or writes, and the authority knows
    /// which version that is.
    #[must_use]
    pub const fn resolve(
        &self,
        domain: ChannelId,
        resource: ResourceKey,
        input_content_version: Option<ContentVersion>,
        output_content_version: Option<ContentVersion>,
    ) -> AccessIntent {
        AccessIntent {
            domain,
            key: match self.extent {
                ParticipationExtent::Range(r) => AccessKey::Range(resource, r),
                ParticipationExtent::Subresource(s) => AccessKey::Subresource(resource, s),
                ParticipationExtent::Whole => AccessKey::Whole(resource),
            },
            mode: self.mode,
            api_stages: self.api_stages,
            input_content_version,
            output_content_version,
        }
    }
}

/// Up to two participations, without an allocation.
///
/// Two, because that is the widest thing any *record* declares by itself: a
/// draw's index buffer and its indirect arguments, a copy's source and its
/// destination, an ICB and its argument buffer. A pass descriptor names more,
/// and that is exactly why it is not a record's own claim — it lives in the
/// transaction's arena and is aggregated in [`crate::exec::ResolvedOperation`],
/// where the arena is in scope.
///
/// An inline array rather than a `Vec` because this is answered once per
/// record of every stream. A heap allocation per record is a cost the shape of
/// the answer does not require, and the two operations that used to return
/// `Vec` were paying it for at most two items.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Participations {
    /// Both slots are always initialized — `len` says how many are the
    /// answer. A slot past `len` is a copy of an earlier one and never read,
    /// which is what lets this be `Copy` with no `Option` per element.
    items: [Option<Participation>; 2],
}

impl Participations {
    /// The record names no memory of its own.
    ///
    /// A real answer and not an absence: every operation class answers this
    /// question, and the classes that touch nothing say so rather than being
    /// skipped by a caller that knows which ones they are.
    pub const NONE: Self = Self { items: [None; 2] };

    #[must_use]
    pub const fn one(a: Participation) -> Self {
        Self {
            items: [Some(a), None],
        }
    }

    #[must_use]
    pub const fn two(a: Participation, b: Participation) -> Self {
        Self {
            items: [Some(a), Some(b)],
        }
    }

    /// One participation, or none, from an `Option` — the shape a record with
    /// a single optional read has.
    #[must_use]
    pub const fn maybe(a: Option<Participation>) -> Self {
        Self { items: [a, None] }
    }

    /// The two optional slots, in record order, with the gaps closed.
    ///
    /// A draw may name arguments and no index buffer, so the slots are
    /// independently optional and the answer must not have a hole in it.
    #[must_use]
    pub fn pair(a: Option<Participation>, b: Option<Participation>) -> Self {
        match (a, b) {
            (Some(a), Some(b)) => Self::two(a, b),
            (Some(only), None) | (None, Some(only)) => Self::one(only),
            (None, None) => Self::NONE,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Participation> {
        self.items.iter().flatten()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.iter().count()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl core::ops::Index<usize> for Participations {
    type Output = Participation;

    /// The `index`th participation, in the order the record names them.
    ///
    /// Indexable because the order is part of the answer — a copy's source is
    /// first and its destination second — and because the slots are packed:
    /// [`Participations::pair`] closes the gap, so a present slot never
    /// follows an absent one and `p[1]` cannot mean "the second slot, which
    /// happens to be empty".
    ///
    /// # Panics
    ///
    /// Past the answer's length, like any slice.
    fn index(&self, index: usize) -> &Participation {
        self.items
            .get(index)
            .and_then(Option::as_ref)
            .expect("participation index past the answer")
    }
}

impl IntoIterator for Participations {
    type Item = Participation;
    type IntoIter = core::iter::Flatten<core::array::IntoIter<Option<Participation>, 2>>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter().flatten()
    }
}

impl Participation {
    /// A read of a buffer window a record named, with the extent it
    /// established.
    ///
    /// `length: None` is a record that named an offset and no size — an
    /// indirect argument block whose layout is not established, say — and it
    /// widens to [`ParticipationExtent::Whole`] rather than narrowing to a
    /// guessed span. The offset is not lost: it is not *carried*, because a
    /// range starting at an offset and running to an unknown end is exactly
    /// the whole resource from the hazard test's point of view, and a shorter
    /// claim is an edge that does not get built, which is a race.
    ///
    /// No shader stage: a record that names a buffer window in its own fields
    /// carries no stage mask. A participation with stages always came from a
    /// record that had them.
    #[must_use]
    pub const fn buffer_read(resource: ResourceId, offset: u64, length: Option<u64>) -> Self {
        Self {
            resource,
            extent: match length {
                Some(length) => ParticipationExtent::Range(ByteRange { offset, length }),
                None => ParticipationExtent::Whole,
            },
            mode: AccessMode::Read,
            api_stages: 0,
        }
    }
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
    /// membership.
    #[test]
    fn a_heap_declaration_meets_its_members_and_not_a_stranger() {
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
        let other_heap = HeapId {
            id: 6,
            membership_generation: 2,
        };
        assert!(
            !AccessKey::Heap(heap).may_alias(AccessKey::Heap(other_heap)),
            "two heaps are two sets of memory"
        );
        assert!(
            !AccessKey::Heap(other_heap).may_alias(member),
            "and a member of one is not a member of the other"
        );
    }

    /// Placing a resource in a heap advances that heap's membership, and the
    /// resource that was already there did not move. A declaration recorded
    /// before the placement and an access recorded after it are talking about
    /// the same bytes, so they must still meet: the generation says which set
    /// was declared, not which memory exists.
    #[test]
    fn a_membership_change_does_not_dissolve_a_heap_hazard() {
        let declared = HeapId {
            id: 5,
            membership_generation: 2,
        };
        let after_a_placement = HeapId {
            id: 5,
            membership_generation: 3,
        };
        assert!(AccessKey::Heap(declared).may_alias(AccessKey::Heap(after_a_placement)));
        let member_now = AccessKey::Whole(ResourceKey {
            backing: BackingId(11),
            heap: Some(after_a_placement),
        });
        assert!(AccessKey::Heap(declared).may_alias(member_now));
        assert!(member_now.may_alias(AccessKey::Heap(declared)));
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
