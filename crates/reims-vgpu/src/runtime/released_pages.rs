//! Guest pages the guest has taken back, and whether this device wrote to one
//! afterwards.
//!
//! # Why this exists, and why [`crate::runtime::node_guard`] is not enough
//!
//! `node_guard` asks whether a host write landed on a page that **was already**
//! a page-table node. On a boot that panicked it read zero, and that result has
//! a blind spot which is now the leading reading of the whole defect:
//!
//! 1. Page `P` is an ordinary data page of task `T`, mapped for a surface.
//! 2. The guest sends `UnmapMemory`, and this device stamps it complete.
//! 3. The guest frees `P` and the allocator hands it back as a **page-table
//!    node** for some other tree.
//! 4. Work this device had in flight over step 1's mapping lands on `P`.
//!
//! `node_guard` first sees `P` as a node at step 3 or later, so the write at
//! step 4 is before its first sighting and reads as `FirstSight` rather than as
//! a finding. The window is exactly the one the guest's ordering rail is
//! supposed to close: it submits the unmap, blocks on the event, and only then
//! runs its own `deallocate` — so **our stamp is what releases that block**, and
//! a stamp written before our work has quiesced is what opens step 4.
//!
//! So this watches the other end. A page released by the guest is a page this
//! device has been told to stop writing, whatever it later becomes, and a write
//! to one is a defect on its own terms — it does not need the page to have
//! become a page table for the answer to be "we wrote where we were told not
//! to". That the corrupting value must be a **zero** word for the guest's
//! assertion to fire is a property of the panic, not of this check.
//!
//! # Terminals, not a horizon
//!
//! A watched page stops being watched when the guest maps it again — at which
//! point writing to it is legitimate, and keeping it would report ordinary work
//! as a defect — or when the task dies. Neither is a duration chosen in
//! advance. That matters here for the same reason it did for
//! [`crate::runtime::objects::slot_recheck`]: a horizon would have to come from
//! somewhere, and the number would end up deciding the answer.
//!
//! # What it costs
//!
//! The page list of a released range, resolved through the run walker that
//! already batches its deepest level, and one lookup per watched page per census
//! sweep. The resolve happens **before** the unmap is applied, which is the only
//! moment those addresses still translate.
//!
//! # What a finding means, and what it does not
//!
//! `released_write` means: between the guest releasing this page and now, this
//! device wrote to it, and the write named its pages. Only
//! [`HostWriteVerdict::Overlap`] counts, for the reason `node_guard` states —
//! this is an alarm, and an alarm that fires on what it cannot decide is worse
//! than no alarm.
//!
//! It does **not** mean the write reached a page table. It means the ordering
//! this device relies on did not hold, which is the precondition for that.

use std::collections::BTreeMap;

use crate::runtime::host_writes::{HostWriteVerdict, HostWrites};

/// How many released pages one task's watch will hold.
///
/// A single release can cover tens of megabytes, so this is a bound on the
/// watch and not on the guest. A page that does not fit is **refused and
/// counted** — see [`ReleasedPages::refused`] — because a quiet drop would
/// shrink the watched population while the readings kept their shape, which is
/// the failure that reads as a clean sweep.
const WATCH_CAP: usize = 4096;

/// What a sweep found for one released page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleasedVerdict {
    /// Nothing this device recorded has touched the page since it was released.
    Quiet,
    /// **This device wrote to a page the guest had taken back.**
    Wrote { since_us: u64 },
    /// A write in the window named no pages, so this page cannot be judged.
    Undecidable,
}

impl ReleasedVerdict {
    /// Whether this is the reading the module exists to find.
    pub fn is_finding(self) -> bool {
        matches!(self, Self::Wrote { .. })
    }

    /// The counter name, one per variant and exhaustive.
    pub fn route(self) -> &'static str {
        match self {
            Self::Quiet => "released_quiet",
            Self::Wrote { .. } => "released_write_after_release",
            Self::Undecidable => "released_undecidable",
        }
    }
}

/// When a page was released.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Released {
    /// The [`HostWrites`] epoch current at the release. Any write carrying a
    /// higher epoch landed after the guest took the page back.
    epoch: u64,
    /// `crate::observe::elapsed_us` at the release.
    at_us: u64,
}

/// One task's released pages.
#[derive(Default, Debug)]
pub struct ReleasedPages {
    pages: BTreeMap<u64, Released>,
    refused: u64,
}

impl ReleasedPages {
    /// Record that the guest has taken `gpa` back.
    ///
    /// A page released twice without an intervening map keeps its **first**
    /// release epoch: the question is whether anything was written since the
    /// guest stopped wanting us there, and re-stamping it would forgive a write
    /// that had already happened.
    pub fn release(&mut self, writes: &HostWrites, gpa: u64, now_us: u64) {
        if self.pages.contains_key(&gpa) {
            return;
        }
        if self.pages.len() >= WATCH_CAP {
            self.refused += 1;
            return;
        }
        self.pages.insert(
            gpa,
            Released {
                epoch: writes.epoch(),
                at_us: now_us,
            },
        );
    }

    /// The guest has mapped `gpa` again, so writing to it is legitimate.
    pub fn remapped(&mut self, gpa: u64) {
        self.pages.remove(&gpa);
    }

    /// Judge every watched page, dropping the ones that answer.
    ///
    /// A page that reports is removed so that one late write is one finding
    /// rather than one per sweep for the rest of the boot. A `Quiet` page stays,
    /// because the write it is waiting for has not happened *yet*.
    pub fn sweep(&mut self, writes: &HostWrites, now_us: u64) -> Vec<(u64, ReleasedVerdict)> {
        let mut out = Vec::new();
        self.pages.retain(|&gpa, rel| {
            let verdict = match writes.wrote_any_since(rel.epoch, &[gpa]) {
                HostWriteVerdict::Quiet => ReleasedVerdict::Quiet,
                HostWriteVerdict::Overlap => ReleasedVerdict::Wrote {
                    since_us: now_us.saturating_sub(rel.at_us),
                },
                _ => ReleasedVerdict::Undecidable,
            };
            out.push((gpa, verdict));
            matches!(verdict, ReleasedVerdict::Quiet)
        });
        out
    }

    /// How many pages are being watched.
    pub fn watched(&self) -> usize {
        self.pages.len()
    }

    /// How many releases this watch turned away because it was full.
    pub fn refused(&self) -> u64 {
        self.refused
    }
}

/// Judge every task's released pages and report the writes that landed after
/// the guest took a page back.
///
/// Runs on the drain tranche, beside
/// [`crate::runtime::objects::slot_recheck::sweep`] and for the same reason: it
/// returns immediately when nothing is watched, which is every rail on which the
/// guest does not release pages under load.
pub fn sweep(state: &mut crate::model::DeviceState) {
    let now_us = crate::observe::elapsed_us();
    let crate::model::DeviceState {
        host_writes,
        released_pages: watches,
        ..
    } = state;
    for (&task_id, watch) in watches.iter_mut() {
        if watch.watched() == 0 {
            continue;
        }
        for (gpa, verdict) in watch.sweep(host_writes, now_us) {
            crate::runtime::drain::note_store_route(verdict.route());
            if let ReleasedVerdict::Wrote { since_us } = verdict {
                if crate::observe::first_sight("released_write_after_release", gpa) {
                    crate::observe::fail(format!(
                        "released_pages reason={} task={task_id} gpa={gpa:#x} \
                         since_us={since_us} watched={} refused={} (this device wrote to a guest \
                         page after the guest released it; the guest is entitled to have given \
                         that page to something else, including its own page table)",
                        verdict.route(),
                        watch.watched(),
                        watch.refused(),
                    ));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = 4096;

    /// A released page nobody writes to stays watched and stays quiet — it is
    /// waiting for a write that may still come.
    #[test]
    fn a_released_page_nobody_wrote_to_stays_quiet_and_stays_watched() {
        let mut r = ReleasedPages::default();
        let writes = HostWrites::default();
        r.release(&writes, 9 * P, 0);
        for t in 1..4 {
            let found = r.sweep(&writes, t);
            assert_eq!(found, vec![(9 * P, ReleasedVerdict::Quiet)]);
            assert_eq!(r.watched(), 1);
        }
    }

    /// A write after the release is the finding, it carries how long after, and
    /// it is reported exactly once.
    #[test]
    fn a_write_after_the_release_is_reported_once() {
        let mut r = ReleasedPages::default();
        let mut writes = HostWrites::default();
        r.release(&writes, 9 * P, 100);
        writes.note_pages(vec![9 * P]);

        let found = r.sweep(&writes, 700);
        assert_eq!(found, vec![(9 * P, ReleasedVerdict::Wrote { since_us: 600 })]);
        assert!(found[0].1.is_finding());
        assert_eq!(r.watched(), 0, "a page that reported is not watched again");
        assert!(r.sweep(&writes, 800).is_empty());
    }

    /// A write *before* the release says nothing — that is ordinary work on a
    /// page the guest still wanted us in.
    #[test]
    fn a_write_before_the_release_is_not_a_finding() {
        let mut r = ReleasedPages::default();
        let mut writes = HostWrites::default();
        writes.note_pages(vec![9 * P]);
        r.release(&writes, 9 * P, 0);
        assert_eq!(r.sweep(&writes, 1), vec![(9 * P, ReleasedVerdict::Quiet)]);
    }

    /// A page the guest maps again leaves the watch, so the writes that follow
    /// are ordinary work and not findings. Without this every recycled page
    /// would report.
    #[test]
    fn a_remapped_page_leaves_the_watch() {
        let mut r = ReleasedPages::default();
        let mut writes = HostWrites::default();
        r.release(&writes, 9 * P, 0);
        r.remapped(9 * P);
        assert_eq!(r.watched(), 0);
        writes.note_pages(vec![9 * P]);
        assert!(r.sweep(&writes, 1).is_empty());
    }

    /// Releasing a page twice keeps the first epoch, so a write that already
    /// happened is not forgiven by the second release.
    #[test]
    fn a_second_release_does_not_forgive_a_write_that_already_landed() {
        let mut r = ReleasedPages::default();
        let mut writes = HostWrites::default();
        r.release(&writes, 9 * P, 0);
        writes.note_pages(vec![9 * P]);
        r.release(&writes, 9 * P, 10);
        assert_eq!(
            r.sweep(&writes, 20),
            vec![(9 * P, ReleasedVerdict::Wrote { since_us: 20 })],
            "the gap is measured from the first release, not the second"
        );
    }

    /// A write that named no pages cannot judge this one, and is not a finding.
    #[test]
    fn an_unnamed_write_is_undecidable() {
        let mut r = ReleasedPages::default();
        let mut writes = HostWrites::default();
        r.release(&writes, 9 * P, 0);
        writes.note_unknown();
        let found = r.sweep(&writes, 1);
        assert_eq!(found, vec![(9 * P, ReleasedVerdict::Undecidable)]);
        assert!(!found[0].1.is_finding());
    }

    /// The watch stops at its capacity and counts what it turned away.
    #[test]
    fn a_full_watch_refuses_and_says_how_often() {
        let mut r = ReleasedPages::default();
        let writes = HostWrites::default();
        for i in 0..WATCH_CAP as u64 {
            r.release(&writes, i * P, 0);
        }
        assert_eq!(r.watched(), WATCH_CAP);
        for i in 0..5u64 {
            r.release(&writes, (WATCH_CAP as u64 + i) * P, 0);
        }
        assert_eq!(r.refused(), 5);
        assert_eq!(r.watched(), WATCH_CAP, "a refusal does not evict");
    }

    /// Every verdict names itself, and exactly one of them is the finding.
    #[test]
    fn every_verdict_names_itself() {
        let all = [
            ReleasedVerdict::Quiet,
            ReleasedVerdict::Wrote { since_us: 0 },
            ReleasedVerdict::Undecidable,
        ];
        let mut names: Vec<&str> = all.iter().map(|v| v.route()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "two verdicts share a route name");
        assert_eq!(all.iter().filter(|v| v.is_finding()).count(), 1);
    }
}
