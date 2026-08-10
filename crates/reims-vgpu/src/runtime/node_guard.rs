//! Whether this device has written into a guest page that holds page-table
//! entries.
//!
//! # Why this exists
//!
//! One guest line keeps a three-level GPU page table whose interior nodes carry
//! both a C++ child-pointer array, in its kernel heap, and a 32-bit PTE word per
//! slot, in a guest page. Its teardown asserts the two agree — a slot with a
//! live child pointer must have a non-zero PTE — and on that line the assertion
//! fires on roughly two boots in five and takes the whole guest with it. The
//! divergence is one-sided: the child pointers are unreachable from here, and
//! the PTE page is ordinary guest RAM that this device reads on every
//! translation and could, if it resolved an address wrongly, write.
//!
//! The interval audit in [`crate::runtime::map_audit`] answered the first
//! question — the guest's map and unmap ranges pair cleanly, including on a boot
//! that panicked — so a pairing bug is not what violates the invariant. This
//! answers the second: does a host write ever land on a node page?
//!
//! **A hit is a proof.** A zero word appearing in a node page is exactly what
//! the guest's assertion reads as a missing entry, and this device writing a
//! page-sized clear or a zero-filled writeback into a recycled page is the
//! PTE-corruption class this repository has already been bitten by once — the
//! map-notify guest flush that `apply_map_family` still carries a comment about.
//!
//! # What it costs, and why it can be always on
//!
//! Nothing is scanned and nothing is walked for its sake. The observation
//! happens where the device already handles a map or unmap packet, which is both
//! the cheapest place to stand and the place the guest is editing the tree; it
//! is one descent, so at most [`MAX_TREE_NODES`] guest reads per lifecycle
//! packet. The query side is a lookup per node page against
//! [`crate::runtime::host_writes`], which is already a complete per-page census
//! of this device's writes and needs nothing added to it.
//!
//! # What a finding means, precisely
//!
//! For a page `g` this device has now seen as a node twice: **between those two
//! sightings, this device wrote to `g`.** Only [`HostWriteVerdict::Overlap`] is
//! reported, so the answer rests on a write that named its pages; the two
//! undecidable verdicts — a write that could not name what it touched, and a
//! record cleared for reaching its bound — are counted separately and are not
//! findings. That is the opposite of how `host_writes` is read by a cache, where
//! an undecidable must be treated as a write, and deliberately so: a cache is
//! deciding whether to trust bytes and must fail safe, while this is an alarm
//! and a false one costs a session.
//!
//! The residual, stated rather than guarded: a page that leaves the tree, is
//! legitimately reused as data, is written, and then returns to the tree would
//! report as a finding. That is a real sequence and this cannot distinguish it,
//! so a finding is the start of an investigation and not a verdict on its own —
//! read the emitted line's `gap_us`, because the innocent version of that story
//! needs a page to make a full round trip.
//!
//! Records are dropped with their task, at the same point
//! [`crate::model::DeviceState`] drops its map intervals: a task's whole address
//! space goes when it does, and a reused id must not inherit a page set that now
//! describes somebody else's memory.

use std::collections::BTreeMap;

use crate::runtime::host_writes::{HostWriteVerdict, HostWrites};

pub use reims_vgpu_paging::resolve::MAX_TREE_NODES;

/// How many distinct node pages one task's watch will hold.
///
/// The guest's tree is depth 3 with a 1024 fanout, so a task mapping a few
/// hundred megabytes populates a root, a handful of mid nodes and one leaf-level
/// node per 4 MiB. This is not sized to hold all of them and does not need to
/// be: the watch only ever learns about nodes on paths the device itself
/// descends, so it is a sample of the tree by construction, and the bound exists
/// so that sample cannot grow without limit on a long boot.
///
/// A page that does not fit is **refused and counted**, never silently dropped —
/// see [`NodeWatch::refused`]. A quiet drop would shrink the watched population
/// while the readings kept their shape, which is the failure mode that reads as
/// a clean sweep.
const WATCH_CAP: usize = 512;

/// What one observation of a node page found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeVerdict {
    /// First sighting of this page as a node. Nothing to compare against yet.
    FirstSight,
    /// This device did not write to the page between two sightings of it as a
    /// node.
    Quiet,
    /// **This device wrote to a page that holds page-table entries.**
    ///
    /// `gap_us` is how long the page went unobserved, which is what says whether
    /// the innocent reading — the page left the tree, served as data, and came
    /// back — had room to happen.
    Wrote { gap_us: u64 },
    /// A write in the window could not be attributed to pages, so the question
    /// cannot be answered for this page. Not a finding.
    Undecidable,
    /// The watch is full and this page is not in it.
    NotWatched,
}

impl NodeVerdict {
    /// Whether this is the reading the module exists to find.
    pub fn is_finding(self) -> bool {
        matches!(self, Self::Wrote { .. })
    }

    /// The counter name, one per variant and exhaustive, so a new verdict
    /// cannot reach a census under a borrowed name.
    pub fn route(self) -> &'static str {
        match self {
            Self::FirstSight => "node_guard_first_sight",
            Self::Quiet => "node_guard_quiet",
            Self::Wrote { .. } => "node_guard_wrote_node_page",
            Self::Undecidable => "node_guard_undecidable",
            Self::NotWatched => "node_guard_not_watched",
        }
    }
}

/// When a node page was last seen as part of the tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sighting {
    /// The [`HostWrites`] epoch current at that sighting. A later write carries
    /// a higher epoch, which is the whole comparison.
    epoch: u64,
    /// `crate::observe::elapsed_us` at that sighting, for [`NodeVerdict::Wrote`].
    at_us: u64,
}

/// One task's watched node pages.
#[derive(Default, Debug)]
pub struct NodeWatch {
    seen: BTreeMap<u64, Sighting>,
    refused: u64,
}

impl NodeWatch {
    /// Observe `gpa` as a node page right now, and say what happened to it since
    /// the last time it was one.
    ///
    /// The sighting is recorded whatever the verdict, so a finding is reported
    /// once per window rather than once per packet for the rest of the boot.
    pub fn observe(&mut self, writes: &HostWrites, gpa: u64, now_us: u64) -> NodeVerdict {
        let epoch = writes.epoch();
        // The capacity is read before the lookup borrows the map: a full watch
        // still answers for a page already in it, and only a page that would
        // have to be *added* is refused.
        let full = self.seen.len() >= WATCH_CAP;
        match self.seen.get_mut(&gpa) {
            Some(prev) => {
                let verdict = match writes.wrote_any_since(prev.epoch, &[gpa]) {
                    HostWriteVerdict::Quiet => NodeVerdict::Quiet,
                    HostWriteVerdict::Overlap => NodeVerdict::Wrote {
                        gap_us: now_us.saturating_sub(prev.at_us),
                    },
                    // Everything `host_writes` cannot decide. Named rather than
                    // folded into either answer: reading it as a write invents
                    // findings and reading it as quiet hides them.
                    _ => NodeVerdict::Undecidable,
                };
                *prev = Sighting {
                    epoch,
                    at_us: now_us,
                };
                verdict
            }
            None if full => {
                self.refused += 1;
                NodeVerdict::NotWatched
            }
            None => {
                self.seen.insert(
                    gpa,
                    Sighting {
                        epoch,
                        at_us: now_us,
                    },
                );
                NodeVerdict::FirstSight
            }
        }
    }

    /// How many distinct pages this watch declined to take because it was full.
    ///
    /// Non-zero means the readings describe a sample smaller than the tree the
    /// device touched, which is a fact about the instrument and belongs next to
    /// its output.
    pub fn refused(&self) -> u64 {
        self.refused
    }

    /// How many pages are being watched.
    pub fn watched(&self) -> usize {
        self.seen.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = 4096;

    /// The first sighting has nothing to compare against, and a page nothing
    /// wrote to stays quiet across any number of sightings.
    #[test]
    fn a_page_this_device_never_wrote_is_quiet_however_often_it_is_seen() {
        let mut w = NodeWatch::default();
        let writes = HostWrites::default();
        assert_eq!(w.observe(&writes, 9 * P, 0), NodeVerdict::FirstSight);
        for t in 1..5 {
            assert_eq!(w.observe(&writes, 9 * P, t), NodeVerdict::Quiet);
        }
        assert_eq!(w.watched(), 1);
    }

    /// A write landing on a watched node page between two sightings of it is
    /// the finding, and it carries how long the gap was.
    #[test]
    fn a_write_between_two_sightings_of_a_node_page_is_the_finding() {
        let mut w = NodeWatch::default();
        let mut writes = HostWrites::default();
        assert_eq!(w.observe(&writes, 9 * P, 100), NodeVerdict::FirstSight);

        writes.note_pages(vec![9 * P]);
        let v = w.observe(&writes, 9 * P, 350);
        assert_eq!(v, NodeVerdict::Wrote { gap_us: 250 });
        assert!(v.is_finding());

        // Recorded at the finding too: the same write must not be reported
        // again at every later sighting.
        assert_eq!(w.observe(&writes, 9 * P, 400), NodeVerdict::Quiet);
    }

    /// A write to some other page says nothing about this one. This is the
    /// property that makes the instrument usable at all — the device writes
    /// guest memory constantly, and only the pages that are tables matter.
    #[test]
    fn a_write_to_a_neighbouring_page_is_not_a_finding() {
        let mut w = NodeWatch::default();
        let mut writes = HostWrites::default();
        w.observe(&writes, 9 * P, 0);
        writes.note_pages(vec![8 * P, 10 * P]);
        assert_eq!(w.observe(&writes, 9 * P, 1), NodeVerdict::Quiet);
    }

    /// A write that could not name its pages is reported as undecidable rather
    /// than as a finding. The alarm direction matters here and it is the
    /// opposite of a cache's.
    #[test]
    fn a_write_that_named_no_pages_is_undecidable_and_not_a_finding() {
        let mut w = NodeWatch::default();
        let mut writes = HostWrites::default();
        w.observe(&writes, 9 * P, 0);
        writes.note_unknown();
        let v = w.observe(&writes, 9 * P, 1);
        assert_eq!(v, NodeVerdict::Undecidable);
        assert!(!v.is_finding());
    }

    /// The watch stops at its capacity and counts what it turned away, rather
    /// than growing without bound or dropping quietly.
    #[test]
    fn a_full_watch_refuses_and_says_how_often() {
        let mut w = NodeWatch::default();
        let writes = HostWrites::default();
        for i in 0..WATCH_CAP as u64 {
            assert_eq!(w.observe(&writes, i * P, 0), NodeVerdict::FirstSight);
        }
        assert_eq!(w.watched(), WATCH_CAP);
        for i in 0..3u64 {
            let gpa = (WATCH_CAP as u64 + i) * P;
            assert_eq!(w.observe(&writes, gpa, 0), NodeVerdict::NotWatched);
        }
        assert_eq!(w.refused(), 3);
        assert_eq!(w.watched(), WATCH_CAP, "a refusal does not evict");
        // A page already in the watch is still answered while it is full.
        assert_eq!(w.observe(&writes, 0, 1), NodeVerdict::Quiet);
    }

    /// Every verdict has its own route name. A census that reused one would
    /// merge two populations and read as a smaller problem than it is.
    #[test]
    fn every_verdict_names_itself() {
        let all = [
            NodeVerdict::FirstSight,
            NodeVerdict::Quiet,
            NodeVerdict::Wrote { gap_us: 0 },
            NodeVerdict::Undecidable,
            NodeVerdict::NotWatched,
        ];
        let mut names: Vec<&str> = all.iter().map(|v| v.route()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "two verdicts share a route name");
        assert_eq!(
            all.iter().filter(|v| v.is_finding()).count(),
            1,
            "exactly one verdict is the finding"
        );
    }
}
