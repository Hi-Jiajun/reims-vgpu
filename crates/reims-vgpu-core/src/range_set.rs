//! A set of byte ranges over one backing, kept sorted and disjoint.
//!
//! # Why the model needs one
//!
//! Content authority is not per resource. A guest writes part of a buffer, a
//! blit fills another part, a draw reads a third — and whether a transfer is
//! owed depends on which *bytes* a replica is behind on, not on whether it is
//! behind at all. A per-resource dirty flag answers the second question and
//! then copies the whole resource to answer the first, which is the cost this
//! representation exists to avoid.
//!
//! # Sorted and disjoint is an invariant, not a convention
//!
//! Every operation restores it before returning, and the tests assert it after
//! each one. An overlapping pair here would make [`RangeSet::subtract`] leave
//! bytes behind, which reads as a transfer that copied everything it was asked
//! to and quietly did not.

use crate::access::ByteRange;

/// A sorted, disjoint, coalesced set of half-open byte ranges.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RangeSet {
    /// Invariant: sorted by offset, non-empty, disjoint, and non-adjacent —
    /// two ranges that touch are one range, so equality is structural.
    ranges: Vec<ByteRange>,
}

impl RangeSet {
    #[must_use]
    pub const fn new() -> Self {
        Self { ranges: Vec::new() }
    }

    #[must_use]
    pub fn from_range(range: ByteRange) -> Self {
        let mut s = Self::new();
        s.insert(range);
        s
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// The ranges, in order.
    #[must_use]
    pub fn ranges(&self) -> &[ByteRange] {
        &self.ranges
    }

    /// Total bytes covered.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.ranges.iter().map(|r| r.length).sum()
    }

    /// Add a range, coalescing with anything it touches.
    pub fn insert(&mut self, range: ByteRange) {
        if range.length == 0 {
            return;
        }
        let end = end_of(range);
        let mut merged = range;
        let mut merged_end = end;
        let mut out: Vec<ByteRange> = Vec::with_capacity(self.ranges.len() + 1);
        let mut placed = false;
        for r in self.ranges.drain(..) {
            let r_end = end_of(r);
            if r_end < merged.offset || merged_end < r.offset {
                // Disjoint and not adjacent. Order decides which side it lands.
                if r_end < merged.offset {
                    out.push(r);
                } else {
                    if !placed {
                        out.push(ByteRange {
                            offset: merged.offset,
                            length: merged_end - merged.offset,
                        });
                        placed = true;
                    }
                    out.push(r);
                }
            } else {
                merged.offset = merged.offset.min(r.offset);
                merged_end = merged_end.max(r_end);
            }
        }
        if !placed {
            out.push(ByteRange {
                offset: merged.offset,
                length: merged_end - merged.offset,
            });
        }
        self.ranges = out;
    }

    /// Remove a range.
    pub fn remove(&mut self, range: ByteRange) {
        if range.length == 0 {
            return;
        }
        let cut_start = range.offset;
        let cut_end = end_of(range);
        let mut out = Vec::with_capacity(self.ranges.len() + 1);
        for r in self.ranges.drain(..) {
            let r_end = end_of(r);
            if r_end <= cut_start || cut_end <= r.offset {
                out.push(r);
                continue;
            }
            if r.offset < cut_start {
                out.push(ByteRange {
                    offset: r.offset,
                    length: cut_start - r.offset,
                });
            }
            if cut_end < r_end {
                out.push(ByteRange {
                    offset: cut_end,
                    length: r_end - cut_end,
                });
            }
        }
        self.ranges = out;
    }

    /// Whether every byte of `range` is covered.
    #[must_use]
    pub fn covers(&self, range: ByteRange) -> bool {
        if range.length == 0 {
            return true;
        }
        self.ranges
            .iter()
            .any(|r| r.offset <= range.offset && end_of(range) <= end_of(*r))
    }

    /// The parts of `range` this set does **not** cover.
    ///
    /// This is the transfer question: given what a replica is fresh for, which
    /// bytes of a read does it still owe?
    #[must_use]
    pub fn missing_from(&self, range: ByteRange) -> RangeSet {
        let mut want = RangeSet::from_range(range);
        for r in &self.ranges {
            want.remove(*r);
        }
        want
    }

    /// Everything in `other` as well as everything here.
    pub fn union_with(&mut self, other: &RangeSet) {
        for r in &other.ranges {
            self.insert(*r);
        }
    }

    /// Drop everything.
    pub fn clear(&mut self) {
        self.ranges.clear();
    }
}

const fn end_of(r: ByteRange) -> u64 {
    r.offset.saturating_add(r.length)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(offset: u64, length: u64) -> ByteRange {
        ByteRange { offset, length }
    }

    /// The invariant every operation restores. Checked after each one rather
    /// than trusted, because an overlapping pair makes `remove` leave bytes
    /// behind — which reads as a transfer that copied everything it was asked
    /// to and quietly did not.
    fn assert_sorted_disjoint_coalesced(s: &RangeSet) {
        for w in s.ranges().windows(2) {
            assert!(
                end_of(w[0]) < w[1].offset,
                "ranges must be ordered, disjoint and non-adjacent: {:?}",
                s.ranges()
            );
        }
        for x in s.ranges() {
            assert!(x.length > 0, "an empty range is not a member");
        }
    }

    #[test]
    fn inserting_touching_ranges_makes_one() {
        let mut s = RangeSet::new();
        s.insert(r(0, 4));
        s.insert(r(4, 4));
        assert_sorted_disjoint_coalesced(&s);
        assert_eq!(s.ranges(), &[r(0, 8)]);
        assert_eq!(s.len(), 8);
    }

    #[test]
    fn inserting_a_gap_keeps_two() {
        let mut s = RangeSet::new();
        s.insert(r(0, 4));
        s.insert(r(8, 4));
        assert_sorted_disjoint_coalesced(&s);
        assert_eq!(s.ranges(), &[r(0, 4), r(8, 4)]);
        // And a range spanning the gap collapses all three.
        s.insert(r(3, 6));
        assert_sorted_disjoint_coalesced(&s);
        assert_eq!(s.ranges(), &[r(0, 12)]);
    }

    #[test]
    fn inserting_out_of_order_still_sorts() {
        let mut s = RangeSet::new();
        for range in [r(100, 4), r(0, 4), r(50, 4), r(20, 4)] {
            s.insert(range);
            assert_sorted_disjoint_coalesced(&s);
        }
        assert_eq!(s.ranges(), &[r(0, 4), r(20, 4), r(50, 4), r(100, 4)]);
    }

    #[test]
    fn removing_the_middle_splits_a_range() {
        let mut s = RangeSet::from_range(r(0, 16));
        s.remove(r(4, 4));
        assert_sorted_disjoint_coalesced(&s);
        assert_eq!(s.ranges(), &[r(0, 4), r(8, 8)]);
    }

    #[test]
    fn removing_the_edges_trims_rather_than_splits() {
        let mut s = RangeSet::from_range(r(8, 8));
        s.remove(r(0, 10));
        assert_eq!(s.ranges(), &[r(10, 6)]);
        s.remove(r(14, 100));
        assert_eq!(s.ranges(), &[r(10, 4)]);
        assert_sorted_disjoint_coalesced(&s);
    }

    #[test]
    fn a_zero_length_range_is_neither_inserted_nor_removed() {
        let mut s = RangeSet::from_range(r(0, 8));
        s.insert(r(4, 0));
        s.remove(r(4, 0));
        assert_eq!(s.ranges(), &[r(0, 8)]);
        assert!(
            s.covers(r(4, 0)),
            "a range naming no byte is always covered"
        );
    }

    /// The transfer question, and the one that has to be exactly right: a
    /// replica fresh for part of a read owes only the rest.
    #[test]
    fn missing_from_names_only_the_bytes_that_are_owed() {
        let mut fresh = RangeSet::new();
        fresh.insert(r(0, 16));
        fresh.insert(r(32, 16));
        let owed = fresh.missing_from(r(0, 64));
        assert_sorted_disjoint_coalesced(&owed);
        assert_eq!(owed.ranges(), &[r(16, 16), r(48, 16)]);
        assert_eq!(owed.len(), 32);
        // Fully covered reads owe nothing.
        assert!(fresh.missing_from(r(0, 16)).is_empty());
        assert!(fresh.covers(r(4, 8)));
        assert!(!fresh.covers(r(8, 32)));
    }

    #[test]
    fn a_read_of_nothing_owes_nothing() {
        let fresh = RangeSet::new();
        assert!(fresh.missing_from(r(0, 0)).is_empty());
        assert_eq!(fresh.missing_from(r(0, 8)).ranges(), &[r(0, 8)]);
    }

    #[test]
    fn union_is_insertion_of_every_member() {
        let mut a = RangeSet::from_range(r(0, 8));
        let mut b = RangeSet::from_range(r(8, 8));
        b.insert(r(100, 4));
        a.union_with(&b);
        assert_sorted_disjoint_coalesced(&a);
        assert_eq!(a.ranges(), &[r(0, 16), r(100, 4)]);
    }

    /// Coalescing makes equality structural, which is what lets a caller ask
    /// "did this change" without walking the set.
    #[test]
    fn two_sets_covering_the_same_bytes_are_equal_however_they_were_built() {
        let mut a = RangeSet::new();
        a.insert(r(0, 4));
        a.insert(r(4, 4));
        let b = RangeSet::from_range(r(0, 8));
        assert_eq!(a, b);
    }
}
