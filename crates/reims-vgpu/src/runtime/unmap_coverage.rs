//! Whether the range the guest is about to tear down still has an entry for
//! every page of it.
//!
//! # The invariant, and whose it is
//!
//! One guest line's GPU page-table teardown walks the `[gva, gva+len)` range it
//! was given one page at a time and refuses, per page, to clear a leaf entry
//! that is **already zero**. The refusal is an assertion: it panics the guest.
//! The same walk refuses to descend through an interior entry that is zero,
//! which is a second assertion at a different site and the same outcome.
//!
//! That teardown runs *after* this device replies to the unmap the guest sent —
//! the guest submits the packet, blocks, and only then edits its own tree. So at
//! the moment the packet arrives, the tree still holds whatever the teardown is
//! about to find, and this device is already entitled to read it. Walking the
//! range here answers whether the guest is about to assert, **before it does**,
//! and names the page it will die on.
//!
//! # Why this is not the same question as the two guards next door
//!
//! [`crate::runtime::node_guard`] and [`crate::runtime::released_pages`] both
//! ask whether this device *wrote* somewhere it should not have. Neither
//! mechanism is required for the assertion above: a range that was torn down
//! twice, or torn down without ever having been built, reaches it with no host
//! write anywhere in the story. Both guards read clean on boots that panicked,
//! and this is the reading that explains how they could.
//!
//! # What the two findings mean
//!
//! - [`Coverage::Absent`] — **no** page of the range has an entry. The range was
//!   already torn down, or was never wired. `level` separates those further: a
//!   zero at the deepest level is a leaf entry that was cleared, and a zero
//!   above it is a subtree that is not there at all.
//! - [`Coverage::Partial`] — some pages have entries and some do not, and
//!   `first_absent` is the page the guest's own walk reaches first and dies on.
//!   A mapping that was only partly wired, or partly torn down.
//!
//! Everything else is counted and silent. [`Coverage::Undecidable`] in
//! particular is **not** a finding: a table page that would not read says
//! nothing about what the guest will find there, and reporting it as absence is
//! how an alarm costs a session for being wrong.
//!
//! # What it costs
//!
//! The walk reuses upper levels across the run and reads the deepest level a
//! batch at a time, so a mapping of `n` pages costs on the order of `n / 64`
//! guest reads rather than `n * depth`. The guest's own teardown walks every one
//! of those pages unconditionally, so the reach asked for is never more than the
//! reach the guest has already committed to.
//!
//! It is gated with the other two guards on [`crate::env::PAGE_GUARDS`], so one
//! switch takes all three out of a boot that is measuring the race they watch.

use reims_vgpu_paging::resolve::RangeCoverage;

/// The most pages one unmap will be walked for.
///
/// **Not a fidelity bound.** The guest walks every page of its own range, so
/// there is no reach this can be too small for in the sense of missing something
/// the guest does not also do. It bounds the drain thread's cost against a
/// length field this device has not validated, which is the only reason it
/// exists: a corrupt or absurd length would otherwise spin the walk for as long
/// as the number says.
///
/// A million pages is four gigabytes at a 4 KiB page and sixteen at 16 KiB —
/// larger than any mapping this device has been observed handed. When it does
/// bite it is **counted**, not silent: see `unmap_coverage_truncated`. A bound
/// that trims a reading without saying so is how a scan reports a clean sweep of
/// a population it could not see.
pub const MAX_SCAN_PAGES: u64 = 1 << 20;

/// Whether the guest guards are observing this boot.
///
/// Shared with [`crate::runtime::node_guard`] deliberately: all three page
/// guards watch the same guest teardown, and an A/B that silences one of them
/// while the others keep running measures neither arm.
pub use crate::runtime::node_guard::enabled;

/// What the range looked like, as a verdict rather than counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coverage {
    /// Every page of the range has an entry. The teardown will not assert.
    Covered,
    /// **No page of the range has an entry.**
    ///
    /// `level` is where the first page's descent stopped, zero-based from the
    /// root; `depth` is the tree's. `level + 1 == depth` is a leaf entry that
    /// reads zero, and anything shallower is an absent subtree.
    Absent { level: u32, depth: u32 },
    /// **Some pages have entries and some do not.**
    ///
    /// `first_absent` is the index within the range of the first page without
    /// one, which is the page the guest's own in-order walk reaches first.
    Partial {
        first_absent: u64,
        absent: u64,
        level: u32,
        depth: u32,
    },
    /// The tree could not be read for any page. Not a finding.
    Undecidable,
    /// The task has no readable root, so there is no range to have an opinion
    /// about. Not a finding.
    Unwalkable,
}

impl Coverage {
    /// Read counts from a walk into the verdict they support.
    ///
    /// Absence outranks undecidability on purpose: a range with one readable
    /// zero entry and a hundred unreadable tables still has a zero entry, and
    /// that zero is what the guest will assert on. The reverse — treating a
    /// range as decided because most of it read — would be the mistake.
    pub fn of(c: &RangeCoverage) -> Self {
        if c.absent == 0 {
            return if c.present == 0 {
                Self::Undecidable
            } else {
                Self::Covered
            };
        }
        if c.present == 0 && c.undecidable == 0 {
            return Self::Absent {
                level: c.first_absent_level,
                depth: c.depth,
            };
        }
        Self::Partial {
            first_absent: c.first_absent_index,
            absent: c.absent,
            level: c.first_absent_level,
            depth: c.depth,
        }
    }

    /// Whether this is a reading the module exists to find.
    pub fn is_finding(self) -> bool {
        matches!(self, Self::Absent { .. } | Self::Partial { .. })
    }

    /// The counter name, one per variant and exhaustive, so a new verdict
    /// cannot reach a census under a borrowed name.
    pub fn route(self) -> &'static str {
        match self {
            Self::Covered => "unmap_coverage_covered",
            Self::Absent { .. } => "unmap_coverage_absent",
            Self::Partial { .. } => "unmap_coverage_partial",
            Self::Undecidable => "unmap_coverage_undecidable",
            Self::Unwalkable => "unmap_coverage_unwalkable",
        }
    }

    /// Whether the zero this found sits at the deepest level of the tree.
    ///
    /// That is the one the observed guest assertion fires on — it refuses to
    /// clear a leaf entry that is already zero. A shallower zero ends the guest
    /// too, at its other assertion, and the two are different defects.
    pub fn is_leaf_level(self) -> Option<bool> {
        match self {
            Self::Absent { level, depth } | Self::Partial { level, depth, .. } => {
                Some(level + 1 == depth)
            }
            _ => None,
        }
    }
}

/// How many pages a range spans, and how many of them will be walked.
///
/// Returns `(spanned, scanned)`. They differ only when [`MAX_SCAN_PAGES`] bites,
/// and a caller that finds them different must say so rather than report the
/// scan as covering the range.
///
/// A length below one page spans no pages: the guest's own teardown returns
/// immediately for one, so there is nothing to predict.
pub fn pages_of(length: u64, page_shift: u32) -> (u64, u64) {
    let page = 1u64 << page_shift;
    if length < page {
        return (0, 0);
    }
    let spanned = length >> page_shift;
    (spanned, spanned.min(MAX_SCAN_PAGES))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(present: u64, absent: u64, undecidable: u64) -> RangeCoverage {
        RangeCoverage {
            pages: present + absent + undecidable,
            present,
            absent,
            undecidable,
            first_absent_index: 3,
            first_absent_level: 2,
            depth: 3,
        }
    }

    /// The four decided shapes each map to their own verdict, and the routes are
    /// distinct.
    #[test]
    fn every_shape_of_counts_reaches_its_own_verdict() {
        assert_eq!(Coverage::of(&counts(4, 0, 0)), Coverage::Covered);
        assert_eq!(
            Coverage::of(&counts(0, 4, 0)),
            Coverage::Absent { level: 2, depth: 3 }
        );
        assert_eq!(
            Coverage::of(&counts(2, 2, 0)),
            Coverage::Partial {
                first_absent: 3,
                absent: 2,
                level: 2,
                depth: 3,
            }
        );
        assert_eq!(Coverage::of(&counts(0, 0, 4)), Coverage::Undecidable);

        let routes = [
            Coverage::Covered.route(),
            Coverage::Absent { level: 2, depth: 3 }.route(),
            Coverage::Partial {
                first_absent: 0,
                absent: 1,
                level: 2,
                depth: 3,
            }
            .route(),
            Coverage::Undecidable.route(),
            Coverage::Unwalkable.route(),
        ];
        for (i, a) in routes.iter().enumerate() {
            for b in &routes[i + 1..] {
                assert_ne!(a, b, "two verdicts share a counter name");
            }
        }
    }

    /// A range that is entirely absent except for tables that would not read is
    /// reported as partial, not as wholly absent.
    ///
    /// The distinction is the whole reason `undecidable` is counted separately:
    /// "the range was torn down twice" is a claim about every page of it, and an
    /// unread table is not evidence for that claim.
    #[test]
    fn an_unread_table_beside_a_zero_entry_downgrades_absent_to_partial() {
        assert!(matches!(
            Coverage::of(&counts(0, 3, 1)),
            Coverage::Partial { .. }
        ));
    }

    /// Only the two shapes that predict a guest assertion are findings.
    #[test]
    fn only_the_two_predictive_shapes_are_findings() {
        assert!(!Coverage::Covered.is_finding());
        assert!(!Coverage::Undecidable.is_finding());
        assert!(!Coverage::Unwalkable.is_finding());
        assert!(Coverage::Absent { level: 2, depth: 3 }.is_finding());
        assert!(Coverage::Partial {
            first_absent: 0,
            absent: 1,
            level: 0,
            depth: 3,
        }
        .is_finding());
    }

    /// The leaf-level question is answered only where there is a zero to ask it
    /// about, and it separates the two guest assertions.
    #[test]
    fn the_leaf_level_question_separates_the_two_assertions() {
        assert_eq!(
            Coverage::Absent { level: 2, depth: 3 }.is_leaf_level(),
            Some(true)
        );
        assert_eq!(
            Coverage::Absent { level: 0, depth: 3 }.is_leaf_level(),
            Some(false)
        );
        assert_eq!(Coverage::Covered.is_leaf_level(), None);
        assert_eq!(Coverage::Undecidable.is_leaf_level(), None);
    }

    /// A range shorter than a page spans nothing, and a range longer than the
    /// walk's reach reports both numbers so the caller can say it was trimmed.
    #[test]
    fn the_page_count_spans_the_range_and_reports_its_own_trim() {
        for shift in [12u32, 14] {
            let page = 1u64 << shift;
            assert_eq!(pages_of(0, shift), (0, 0));
            assert_eq!(pages_of(page - 1, shift), (0, 0));
            assert_eq!(pages_of(page, shift), (1, 1));
            assert_eq!(pages_of(page * 5 + 7, shift), (5, 5));

            let over = (MAX_SCAN_PAGES + 9) << shift;
            assert_eq!(pages_of(over, shift), (MAX_SCAN_PAGES + 9, MAX_SCAN_PAGES));
        }
    }
}
