//! Which guest pages this device has written, and when.
//!
//! The hypervisor's dirty bitmap witnesses guest CPU stores and nothing else, so
//! a host-side copy vouched for by "the guest has not written since I looked" can
//! still be stale — because *we* wrote. This is the missing half of that witness.
//!
//! # Why pages and not mappings
//!
//! Three candidate rules were scored against a full content fold before this one
//! was built, each by its own census counter. Those counters are gone with the
//! rules they scored — [`crate::runtime::gather_witness`] takes only the page-exact answer
//! now — so the names below are what the readings were called at the time and are
//! not greppable in a current log.
//!
//! A per-mapping count was measured first and it leaks. One driven boot read
//! fifteen binds where the sampled window's own mapping had not been written, the
//! guest had not written, and the bytes moved anyway. Guest pages are reachable
//! under more than one mapping id, so "mapping 12 was not written" is not "these pages were not
//! written", and a cache keyed on the former serves stale pixels fifteen times a
//! minute.
//!
//! The same boot read zero counterexamples for a *global* count, which moves for
//! every write anywhere — because it moves for every write including the ones a
//! narrower rule fails to attribute. Sound, and it invalidates a texture because
//! an unrelated scanout was composited.
//!
//! # Where it stands
//!
//! Once every writer records here — which took a second pass, because
//! `map_fresh_span_within`'s callers write through a raw alias and were invisible
//! to a hand-picked list of call sites — a driven boot reads **zero** binds where
//! the page-exact rule vouched and the bytes had moved, alongside zero for both
//! wider rules. Of the binds where the guest was quiet and the bytes were
//! identical, this rule serves 93 %; the rest are windows whose page set had just
//! moved.
//!
//! That measurement is what the fold is still there for. It runs on one bind in
//! [`crate::runtime::gather_witness::AUDIT_STRIDE`] rather than all of them, and its
//! counterexample cell is `gw_audit_unsound`: a standing alarm on the rule this
//! module exists to make sound, rather than the per-bind decision it began as.
//!
//! What that licenses is a cache over the zero-copy sampled gathers, valid iff
//! the hypervisor's guest generation has not moved **and** this says the pages
//! were not written. Neither half is sufficient alone and the measurements above
//! are what say so, rather than an argument that they ought to be.
//!
//! Built and measured live on a driven x86/PCI boot: **5852 gathers skipped
//! against 4167 taken, 14.25 GB not read against 4.56 GB read — 75.8 % of the
//! rail's bytes gone** — with all three unsound cells still zero and a Wikipedia
//! page rendering correctly under scroll.
//!
//! # Shape
//!
//! A ring of recent writes rather than a per-page map. A per-page map costs the
//! writer one insert per page written — a 1920x1080 scanout is ~2000 of them — and
//! the reader one lookup per page read, on every bind. The ring costs the writer
//! O(1) and costs the reader nothing at all in the common case, because between
//! two binds of the same window (~8 ms apart) there is usually no host write to
//! compare against.
//!
//! Everything here fails closed. A write that cannot name its pages, a ring that
//! has dropped the entry a reader is asking about, and a mapping re-pointed since
//! the write that named it all answer "assume written".

use std::collections::BTreeSet;

/// What one host write touched.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Wrote {
    /// Every page of a mapping, as its page list stood at `map_generation`.
    ///
    /// The pages are resolved at read time rather than copied at write time,
    /// which is what makes recording a write O(1). `map_generation` is what makes
    /// that safe: a mapping re-pointed since the write has a different page list,
    /// and testing against the new one would answer about pages the write never
    /// touched.
    Mapping { mid: u32, map_generation: u32 },
    /// An explicit page set, for writers that walk the guest page tables and so
    /// name no mapping at all.
    Pages(Vec<u64>),
    /// A writer that could not say. Every reader older than this must assume its
    /// pages were among them.
    Unknown,
}

/// Why [`HostWrites::wrote_any_since`] could not call a window quiet.
///
/// Four causes used to share one `true`, and only the first of them means the
/// bytes under the reader's pages actually moved. The other three are this
/// type's fail-closed rule firing — a correct answer to "can you rule this out",
/// and a very different thing to report. A boot that reads mostly `Unnamed` says
/// its writers are not naming their pages; one that reads mostly `Aged` says
/// [`RING`] is too small for the write rate; one that reads mostly `Overlap`
/// says the device really is writing the windows it samples, and only then is
/// there nothing to reclaim here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostWriteVerdict {
    /// Nothing this device recorded touched these pages.
    Quiet,
    /// A recorded write names one of these pages — the bytes moved.
    Overlap,
    /// A writer that could not say which pages it landed in, so every reader
    /// older than it must assume its own.
    Unnamed,
    /// The ring no longer holds the writes this reader is asking about, so it
    /// cannot be told that nothing touched it.
    Aged,
    /// A mapping-named write whose page list can no longer be reconstructed:
    /// the mapping is gone, or has been re-pointed since the write named it.
    Unresolvable,
}

impl HostWriteVerdict {
    /// True for everything except [`Self::Quiet`] — the sense the caller needs
    /// when it is deciding whether to vouch, spelled once so a new variant
    /// cannot be silently read as quiet.
    pub fn wrote(self) -> bool {
        !matches!(self, Self::Quiet)
    }

    /// Census route naming this verdict, for the witness that reports it.
    pub fn route(self) -> &'static str {
        match self {
            Self::Quiet => "gw_hw_quiet",
            Self::Overlap => "gw_hw_overlap",
            Self::Unnamed => "gw_hw_unnamed",
            Self::Aged => "gw_hw_aged",
            Self::Unresolvable => "gw_hw_unresolvable",
        }
    }
}

/// One retained write, with the digest that lets [`HostWrites::push`] find an
/// identical earlier one without comparing every page list it holds.
#[derive(Debug)]
struct Recent {
    epoch: u64,
    digest: u64,
    what: Wrote,
}

/// A cheap equality filter over [`Wrote`]. Not a hash of anything persisted, so
/// its only requirement is that equal values agree — inequality is confirmed by
/// the full comparison beside it.
fn digest_of(what: &Wrote) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    match what {
        Wrote::Mapping {
            mid,
            map_generation,
        } => (0u8, mid, map_generation).hash(&mut h),
        Wrote::Pages(pages) => (1u8, pages).hash(&mut h),
        Wrote::Unknown => 2u8.hash(&mut h),
    }
    h.finish()
}

/// A per-page last-write record, kept **beside** the ring and never consulted
/// for a decision.
///
/// # Why this exists at all, when the module doc rejected it
///
/// This module's `# Shape` section rejected a per-page map on the grounds that
/// "the ring costs the writer O(1) and costs the reader nothing at all in the
/// common case, because between two binds of the same window (~8 ms apart) there
/// is usually no host write to compare against". [`RING`]'s own doc then records
/// that the premise does not survive compositing, and a driven macos-26 boot puts
/// a number on it:
///
/// ```text
/// gw_hw_quiet     16 402      nothing wrote it
/// gw_hw_aged       6 294      the ring no longer knows
/// gw_hw_overlap       14      something actually wrote it
/// ```
///
/// Fourteen. In a whole boot this device had really written the window it was
/// asked about fourteen times, and refused 6 294 asks because the ring had
/// forgotten — 450 to one — at a cost of 17.2 GB of guest memory re-read.
///
/// A page-keyed map answers every one of those exactly, in `O(pages)` with no
/// horizon. The writer cost the doc worried about is real (~2 000 inserts for a
/// 1080p scanout) and is paid ~28 times a second, against ~2 800 binds a second
/// each looking up ~48 pages.
///
/// # It decides nothing yet, on purpose
///
/// A false quiet here serves stale pixels — the module doc names a black screen
/// that came from exactly that. So this runs as a shadow and only *reports*
/// whether it agrees with the ring. The reading that licenses a swap is
/// `hw_shadow_would_vouch` large and **`hw_shadow_unsound` zero**: the second is
/// every ask where the ring said the pages were written and this would have
/// called them quiet, which is the only direction that can lose a frame.
#[derive(Default, Debug)]
struct PageEpochs {
    /// Last epoch at which this device wrote each page it has ever written.
    last_write: std::collections::HashMap<u64, u64>,
    /// Newest write that could not name its pages. Fail-closed for any reader
    /// older than it, exactly as the ring's `Unknown` entry is.
    unnamed_at: u64,
    /// Newest epoch at which the map was cleared for reaching [`Self::PAGES`].
    ///
    /// Clearing is the only sound way to shed entries: forgetting *one* page
    /// would answer `Quiet` for a page that was written, which is the direction
    /// that loses frames. So the whole map goes and every reader older than the
    /// reset is refused, which is the ring's own failure mode confined to a rare
    /// event instead of a per-ask one.
    reset_at: u64,
}

impl PageEpochs {
    /// The most pages the map will hold before it resets.
    ///
    /// A 1080p scanout is ~2 000 pages, so this is a few hundred distinct
    /// full-screen surfaces' worth. Sized to make `hw_shadow_reset` a rare event
    /// rather than to fit a measurement — if it is not rare, that counter says
    /// so and the shadow's own verdict degrades to the ring's.
    const PAGES: usize = 1 << 20;

    fn note_pages(&mut self, pages: &[u64], epoch: u64) {
        if self.last_write.len().saturating_add(pages.len()) > Self::PAGES {
            self.last_write.clear();
            self.reset_at = epoch;
            crate::runtime::drain::note_store_route("hw_shadow_reset");
        }
        for &p in pages {
            self.last_write.insert(p, epoch);
        }
    }

    fn note_unknown(&mut self, epoch: u64) {
        self.unnamed_at = epoch;
    }

    /// The verdict this map would give, in the ring's own vocabulary.
    fn verdict(&self, since: u64, pages: &[u64]) -> HostWriteVerdict {
        if since < self.unnamed_at {
            return HostWriteVerdict::Unnamed;
        }
        if since < self.reset_at {
            return HostWriteVerdict::Aged;
        }
        if pages
            .iter()
            .any(|p| self.last_write.get(p).is_some_and(|&e| e > since))
        {
            return HostWriteVerdict::Overlap;
        }
        HostWriteVerdict::Quiet
    }
}

/// Compare the shadow against the ring's decision and count the four outcomes
/// that matter.
///
/// The names say what a swap would do, not which values differ, because "they
/// disagree" is not actionable in either direction on its own.
fn note_shadow(ring: HostWriteVerdict, shadow: HostWriteVerdict) {
    crate::runtime::drain::note_store_route(shadow_route(ring, shadow));
}

/// The banding, split from the emit so the one judgement a swap turns on is
/// testable without a device or a guest.
fn shadow_route(ring: HostWriteVerdict, shadow: HostWriteVerdict) -> &'static str {
    match (ring.wrote(), shadow.wrote()) {
        // The whole prize: the ring refused and the map can rule the window out.
        (true, false) if ring != HostWriteVerdict::Overlap => "hw_shadow_would_vouch",
        // The ring said the bytes really moved and the map would have vouched.
        // **This is the only unsound direction** — a frame is lost for each one,
        // and a swap needs this at zero across boots.
        (true, false) => "hw_shadow_unsound",
        // The map is stricter than the ring. Costs a re-gather, loses nothing.
        (false, true) => "hw_shadow_stricter",
        _ => "hw_shadow_agree",
    }
}

/// Recent host writes into guest RAM, newest last.
#[derive(Default, Debug)]
pub struct HostWrites {
    /// Monotonic stamp; a reader records the value current at its own read and
    /// asks later whether anything newer touched its pages. Never 0 once any
    /// write has happened, so 0 is usable as "never looked".
    epoch: u64,
    /// The page-keyed record that would replace `recent`, reporting only.
    /// See [`PageEpochs`] for what has to read zero before it may decide.
    shadow: PageEpochs,
    recent: std::collections::VecDeque<Recent>,
    /// Oldest mark the ring can still answer for.
    ///
    /// A reader with mark `s` asks about writes with epoch **greater than** `s`,
    /// so the ring can answer it exactly when it holds every such write — that
    /// is, when `s` is at least one below the oldest epoch retained. A reader
    /// below that is asking about writes that have been dropped, and gets
    /// "assume written".
    answers_from: u64,
}

/// How many writes the ring remembers.
///
/// Bounds the reader's scan, not memory: a `Mapping` entry is two words and
/// resolves its pages on demand. Sized well above the number of host writes that
/// can fall between two binds of the same sampled window — one driven boot read
/// ~28 host writes a second against ~330 gathers a second, so the usual answer is
/// zero entries to scan and the tail is single digits.
///
/// # That sizing does not survive compositing, and this is the rail's largest cost
///
/// A driven x86/PCI Safari drag reads
/// [`HostWriteVerdict::Aged`] **4275** times against
/// [`HostWriteVerdict::Overlap`] **5**, out of 9986 asks. So 43 % of every
/// question [`crate::runtime::gather_witness`] puts to this record is refused
/// because the ring no longer holds the writes being asked about — not because
/// anything wrote the window — and each refusal costs that window a full
/// re-gather out of guest RAM. The paragraph above is what a quiet workload
/// measures; under compositing the write rate between two binds of one window
/// exceeds this.
///
/// # The reach was banded, and the answer was not "make it bigger"
///
/// [`reach_band`] took that distribution on a driven drag. Of 5 666 asks,
/// 1 362 were inside the ring and **4 294 sat in `hw_reach_le16x`** — reach
/// between 4x and 16x this bound. Three asks in the whole boot reached further.
///
/// A 16x ring would answer them and would be the wrong repair, because the scan
/// is the reach: `wrote_any_since` walks back to the reader's mark, and a
/// `Pages` entry for a 1920x1080 frame holds ~2 000 addresses to test. Sixteen
/// times the bound is sixteen times the walk on exactly the asks that reach
/// furthest, which is the asks that matter.
///
/// The reading says something more useful than a size. A compositor re-Stores
/// the **same page set** every frame, so those 256-to-1024-deep reaches are a
/// handful of distinct surfaces written over and over — and an older write to a
/// page set a newer write already covers rules out nothing the newer one does
/// not. So [`HostWrites::push`] supersedes: an incoming write equal to one the
/// ring holds replaces it in place instead of appending, and this bound counts
/// **distinct** writes rather than write events. A drag's live surfaces fit,
/// and `answers_from` stops moving.
///
/// That is why the number did not change. What changed is what it counts.
const RING: usize = 64;

/// Census route banding how far back one ask reaches, in ring entries.
///
/// This is the distribution [`RING`]'s doc asks for, and the bands are multiples
/// of [`RING`] rather than round numbers so a reading answers the sizing
/// question directly: everything in `hw_reach_le4x` is a question a ring four
/// times this one would have answered, and everything in `hw_reach_over64x` is
/// one no affordable ring reaches, which is the reading that says the repair is
/// writers naming their pages instead.
///
/// The first band is `<= RING` and not `< RING`: a reader whose mark is exactly
/// [`RING`] entries back is asking about the [`RING`] writes the full ring still
/// holds, so it is answerable. Below that boundary `Aged` cannot occur, and a
/// non-zero `hw_reach_in_ring` beside a non-zero `gw_hw_aged` would mean the
/// ring is being trimmed by something other than its own bound.
fn reach_band(reach: u64) -> &'static str {
    const R: u64 = RING as u64;
    match reach {
        r if r <= R => "hw_reach_in_ring",
        r if r <= R * 4 => "hw_reach_le4x",
        r if r <= R * 16 => "hw_reach_le16x",
        r if r <= R * 64 => "hw_reach_le64x",
        _ => "hw_reach_over64x",
    }
}

impl HostWrites {
    /// The stamp a reader records beside a copy it has just taken.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Record a write covering every page of `mapping_id`, as its page list
    /// stands now.
    ///
    /// `pages` is that same page list resolved by the caller, which only the
    /// page-keyed record uses — the ring resolves the mapping again at read
    /// time, and that difference is the point. `None` means the caller could not
    /// name every page, which is recorded as an unnamed write rather than as a
    /// partial one.
    ///
    /// **Both records are fed here and there is no way to feed one alone.** The
    /// two used to be separate calls, and a writer that reached the ring without
    /// the map would have made the map answer `Quiet` for a page that was
    /// written — an omission no comparison could catch, because the ring would
    /// not have known either and the pair would have agreed.
    pub fn note_mapping(&mut self, mapping_id: u32, map_generation: u32, pages: Option<&[u64]>) {
        self.push(Wrote::Mapping {
            mid: mapping_id,
            map_generation,
        });
        match pages {
            Some(p) => self.shadow.note_pages(p, self.epoch),
            None => self.shadow.note_unknown(self.epoch),
        }
    }

    /// Record a write covering exactly `pages` (page-aligned guest addresses).
    pub fn note_pages(&mut self, pages: Vec<u64>) {
        self.shadow.note_pages(&pages, self.epoch.wrapping_add(1));
        self.push(Wrote::Pages(pages));
    }

    /// Record a write whose pages are not known. Invalidates every reader older
    /// than it.
    pub fn note_unknown(&mut self) {
        self.shadow.note_unknown(self.epoch.wrapping_add(1));
        self.push(Wrote::Unknown);
    }

    fn push(&mut self, what: Wrote) {
        self.epoch = self.epoch.wrapping_add(1);
        // Supersede an identical write rather than appending beside it. An older
        // write to a page set this one repeats rules out nothing the newer one
        // does not — for any mark `s`, the newer entry answers every reader the
        // older would have — so dropping it costs no precision and stops a
        // compositor's per-frame re-Store of one surface from spending the whole
        // bound. This is what makes [`RING`] a count of distinct writes.
        //
        // The digest is only a filter: equal digests still compare in full, so a
        // collision costs a `Vec` comparison and never merges two page sets that
        // differ.
        let digest = digest_of(&what);
        if let Some(at) = self
            .recent
            .iter()
            .position(|e| e.digest == digest && e.what == what)
        {
            self.recent.remove(at);
        }
        self.recent.push_back(Recent {
            epoch: self.epoch,
            digest,
            what,
        });
        // An eviction is the only thing that really loses a write, so it is the
        // only thing that may move `answers_from`. Taking it from the evicted
        // entry rather than from the new front is what keeps a superseded entry
        // — which leaves a gap in the retained epochs — from reading as a drop.
        while self.recent.len() > RING {
            if let Some(dropped) = self.recent.pop_front() {
                self.answers_from = self.answers_from.max(dropped.epoch);
            }
        }
    }

    /// Has this device written any of `pages` since `since`, and if so on what
    /// grounds?
    ///
    /// `since` is a value previously returned by [`Self::epoch`]. Everything it
    /// cannot decide answers as written, and names which of the three
    /// undecidables it was: a dropped ring entry, an unnamed write, or a mapping
    /// whose page list has moved since the write named it. Only
    /// [`HostWriteVerdict::Overlap`] says a recorded write actually covers one
    /// of `pages`.
    pub fn wrote_any_since(
        &self,
        state: &crate::model::DeviceState,
        since: u64,
        pages: &[u64],
    ) -> HostWriteVerdict {
        crate::runtime::drain::note_store_route(reach_band(self.epoch.saturating_sub(since)));
        let ring = self.ring_verdict(state, since, pages);
        let exact = self.shadow.verdict(since, pages);
        note_shadow(ring, exact);
        // The page-keyed record decides. The ring still runs and is still
        // compared, so `hw_shadow_unsound` remains a live alarm on the answer
        // being given rather than a retired one — it now reads "this answer
        // vouched and the ring says the bytes really moved", which is the same
        // losing direction it always named.
        exact
    }

    /// The ring's own answer — the one that decides, unchanged.
    ///
    /// Split from [`Self::wrote_any_since`] so the shadow can be compared against
    /// it without the comparison being able to alter it: this function does not
    /// see the shadow and the caller cannot reach past it.
    fn ring_verdict(
        &self,
        state: &crate::model::DeviceState,
        since: u64,
        pages: &[u64],
    ) -> HostWriteVerdict {
        if since < self.answers_from {
            return HostWriteVerdict::Aged;
        }
        let mut asked: Option<BTreeSet<u64>> = None;
        for Recent { epoch, what, .. } in self.recent.iter().rev() {
            if *epoch <= since {
                break;
            }
            let want = asked.get_or_insert_with(|| pages.iter().copied().collect());
            match what {
                Wrote::Unknown => return HostWriteVerdict::Unnamed,
                Wrote::Pages(written) => {
                    if written.iter().any(|p| want.contains(p)) {
                        return HostWriteVerdict::Overlap;
                    }
                }
                Wrote::Mapping {
                    mid,
                    map_generation,
                } => {
                    let Some(m) = state.mappings.get(mid) else {
                        // The mapping is gone, so its page list cannot be
                        // reconstructed to be ruled out.
                        return HostWriteVerdict::Unresolvable;
                    };
                    if m.map_generation != *map_generation {
                        return HostWriteVerdict::Unresolvable;
                    }
                    let shift = state.page_shift;
                    if m.page_entries.iter().any(|&e| {
                        crate::contract::iosurface_pages::entry_gpa_shift(e, shift)
                            .is_some_and(|gpa| want.contains(&gpa))
                    }) {
                        return HostWriteVerdict::Overlap;
                    }
                }
            }
        }
        HostWriteVerdict::Quiet
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = 4096;

    /// The shadow must agree with the ring on the one thing the ring is good at:
    /// a page it holds a write for is not quiet, and one it does not is.
    #[test]
    fn the_shadow_rules_a_window_in_and_out_by_page() {
        let mut s = PageEpochs::default();
        s.note_pages(&[9 * P, 10 * P], 1);
        assert_eq!(s.verdict(0, &[3 * P, 4 * P]), HostWriteVerdict::Quiet);
        assert_eq!(s.verdict(0, &[4 * P, 10 * P]), HostWriteVerdict::Overlap);
        // And a reader that already saw the write is not invalidated by it.
        assert_eq!(s.verdict(1, &[10 * P]), HostWriteVerdict::Quiet);
    }

    /// The shadow has no horizon: a reader whose mark is arbitrarily old still
    /// gets an exact answer. This is the whole reason it exists — the ring
    /// answers `Aged` here, 6 294 times a driven macos-26 boot.
    #[test]
    fn the_shadow_answers_a_reader_the_ring_would_have_forgotten() {
        let mut s = PageEpochs::default();
        for epoch in 1..(RING as u64 * 64) {
            s.note_pages(&[(epoch + 100) * P], epoch);
        }
        // A mark from before every one of those writes, asking about a page none
        // of them touched.
        assert_eq!(s.verdict(0, &[7 * P]), HostWriteVerdict::Quiet);
        // And one they did.
        assert_eq!(s.verdict(0, &[101 * P]), HostWriteVerdict::Overlap);
    }

    /// An unnamed write invalidates every older reader, exactly as the ring's
    /// `Unknown` entry does. Fail-closed is the only direction that is safe.
    #[test]
    fn the_shadow_fails_closed_on_a_write_that_named_no_pages() {
        let mut s = PageEpochs::default();
        s.note_unknown(5);
        assert_eq!(s.verdict(4, &[7 * P]), HostWriteVerdict::Unnamed);
        assert_eq!(s.verdict(5, &[7 * P]), HostWriteVerdict::Quiet);
    }

    /// Shedding entries one page at a time would answer `Quiet` for a page that
    /// was written, which is the direction that loses a frame. The map clears
    /// whole and refuses everything older instead.
    #[test]
    fn a_full_shadow_clears_whole_and_refuses_every_older_reader() {
        let mut s = PageEpochs::default();
        let pages: Vec<u64> = (0..PageEpochs::PAGES as u64).map(|i| i * P).collect();
        s.note_pages(&pages, 1);
        assert_eq!(s.verdict(0, &[3 * P]), HostWriteVerdict::Overlap);
        // The next write does not fit, so the map resets rather than evicting.
        s.note_pages(&[u64::from(u32::MAX) * P], 2);
        assert_eq!(s.verdict(0, &[3 * P]), HostWriteVerdict::Aged);
        assert_eq!(s.verdict(2, &[3 * P]), HostWriteVerdict::Quiet);
    }

    /// The comparison must name a swap's consequence, not which values differ.
    /// The unsound cell is the one a swap gates on and it is a single case:
    /// the ring saw a real write and the shadow would have vouched.
    #[test]
    fn the_shadow_comparison_separates_the_prize_from_the_one_unsafe_direction() {
        use HostWriteVerdict::*;
        // The prize: the ring refused for a reason that is not a real write.
        for refused in [Aged, Unnamed, Unresolvable] {
            assert_eq!(shadow_route(refused, Quiet), "hw_shadow_would_vouch");
        }
        // The only direction that loses a frame.
        assert_eq!(shadow_route(Overlap, Quiet), "hw_shadow_unsound");
        // Stricter costs a re-gather and loses nothing.
        assert_eq!(shadow_route(Quiet, Overlap), "hw_shadow_stricter");
        assert_eq!(shadow_route(Quiet, Quiet), "hw_shadow_agree");
        assert_eq!(shadow_route(Overlap, Overlap), "hw_shadow_agree");
        assert_eq!(shadow_route(Aged, Unnamed), "hw_shadow_agree");
    }

    #[test]
    fn a_write_to_other_pages_leaves_this_window_quiet() {
        let state =
            crate::model::DeviceState::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_X86);
        let mut w = HostWrites::default();
        let mark = w.epoch();
        w.note_pages(vec![9 * P, 10 * P]);
        assert_eq!(
            w.wrote_any_since(&state, mark, &[3 * P, 4 * P]),
            HostWriteVerdict::Quiet
        );
        assert_eq!(
            w.wrote_any_since(&state, mark, &[4 * P, 10 * P]),
            HostWriteVerdict::Overlap
        );
    }

    /// The reader asks about writes *after* its own mark, so the write it
    /// already accounted for must not invalidate it forever.
    #[test]
    fn a_write_the_reader_already_saw_does_not_answer_for_a_later_mark() {
        let state =
            crate::model::DeviceState::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_X86);
        let mut w = HostWrites::default();
        w.note_pages(vec![4 * P]);
        let after = w.epoch();
        assert_eq!(
            w.wrote_any_since(&state, after, &[4 * P]),
            HostWriteVerdict::Quiet
        );
        w.note_pages(vec![4 * P]);
        assert_eq!(
            w.wrote_any_since(&state, after, &[4 * P]),
            HostWriteVerdict::Overlap
        );
    }

    /// A write that could not name its pages must invalidate everything, and a
    /// reader older than the ring must be told **the ring** cannot answer.
    ///
    /// The horizon half asserts `ring_verdict`, not `wrote_any_since`. The ring
    /// no longer decides — the page-keyed record does, and it answers this reader
    /// exactly, which is the whole point of the swap. The ring's own contract is
    /// still tested because the ring is still computed and still compared, and
    /// `hw_shadow_unsound` is only a live alarm for as long as that stays true.
    #[test]
    fn what_the_ring_cannot_decide_reads_as_written() {
        let state =
            crate::model::DeviceState::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_X86);
        let mut w = HostWrites::default();
        let mark = w.epoch();
        w.note_unknown();
        assert_eq!(
            w.wrote_any_since(&state, mark, &[999 * P]),
            HostWriteVerdict::Unnamed,
            "a writer that named no pages must be reported as unnamed and not as \
             a write that landed in this window"
        );

        let mut w = HostWrites::default();
        let stale = w.epoch();
        for i in 0..(RING as u64 + 5) {
            w.note_pages(vec![(100 + i) * P]);
        }
        assert_eq!(
            w.ring_verdict(&state, stale, &[3 * P]),
            HostWriteVerdict::Aged,
            "a mark older than the ring must not be answered from what is left of it"
        );
        assert_eq!(
            w.wrote_any_since(&state, stale, &[3 * P]),
            HostWriteVerdict::Quiet,
            "and the record that decides now answers that same reader exactly"
        );
        let fresh = w.epoch();
        assert_eq!(
            w.wrote_any_since(&state, fresh, &[3 * P]),
            HostWriteVerdict::Quiet
        );
    }

    /// A compositor re-Storing one surface every frame must not spend the bound,
    /// and the repeat must still answer for itself.
    ///
    /// This is the reading the whole change is for: before superseding, the 4 097
    /// repeats below pushed the first write out of the ring and every reader
    /// older than the last [`RING`] frames read [`HostWriteVerdict::Aged`].
    #[test]
    fn a_write_repeated_every_frame_occupies_one_entry_and_never_ages() {
        let state =
            crate::model::DeviceState::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_X86);
        let mut w = HostWrites::default();
        let surface = vec![10 * P, 11 * P];
        let stale = w.epoch();
        for _ in 0..(RING as u64 * 64 + 1) {
            w.note_pages(surface.clone());
        }

        assert_eq!(w.recent.len(), 1, "one distinct write, one entry");
        assert_eq!(
            w.wrote_any_since(&state, stale, &[999 * P]),
            HostWriteVerdict::Quiet,
            "a mark from before every one of those frames is still answerable"
        );
        // And the surviving entry answers for the whole run it stands in for, not
        // only for the last frame — dropping the older repeats must lose no reach.
        assert_eq!(
            w.wrote_any_since(&state, stale, &[11 * P]),
            HostWriteVerdict::Overlap
        );

        // A second distinct surface takes its own entry, so superseding merges
        // repeats and not neighbours.
        w.note_pages(vec![20 * P]);
        assert_eq!(w.recent.len(), 2);
        assert_eq!(
            w.wrote_any_since(&state, stale, &[10 * P]),
            HostWriteVerdict::Overlap,
            "the first surface is still named after another was written"
        );
    }

    /// Superseding leaves gaps in the retained epochs, so `answers_from` may only
    /// move when an entry is really evicted. Taken from the *new front* instead,
    /// it would jump to the newest superseded epoch and refuse readers the ring
    /// can still answer — the failure that reads as "aging got worse".
    #[test]
    fn superseding_does_not_move_the_mark_the_ring_answers_from() {
        let state =
            crate::model::DeviceState::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_X86);
        let mut w = HostWrites::default();
        let stale = w.epoch();
        w.note_pages(vec![P]);
        // Repeats of a *later* set, each superseding the previous, so the front
        // stays the epoch-1 entry while the epochs behind it become sparse.
        for _ in 0..(RING as u64 * 4) {
            w.note_pages(vec![2 * P]);
        }
        assert_eq!(w.recent.len(), 2);
        assert_eq!(
            w.wrote_any_since(&state, stale, &[3 * P]),
            HostWriteVerdict::Quiet
        );
        assert_eq!(
            w.wrote_any_since(&state, stale, &[P]),
            HostWriteVerdict::Overlap
        );
    }

    /// The band that reads "the ring answered this" has to end exactly where the
    /// ring stops answering, or a boot's reading says a bigger ring is needed by
    /// a margin that is really an off-by-one in the census.
    ///
    /// Walks the boundary from both sides against the live [`HostWrites`] rather
    /// than against a second copy of the arithmetic, so the two cannot drift.
    #[test]
    fn the_first_reach_band_ends_where_the_ring_stops_answering() {
        let state =
            crate::model::DeviceState::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_X86);
        let mut w = HostWrites::default();
        // Fill well past the bound so `answers_from` is set by eviction and not
        // by an empty ring.
        for i in 0..(RING as u64 * 3) {
            w.note_pages(vec![(100 + i) * P]);
        }
        let now = w.epoch();
        for reach in 0..=(RING as u64 + 2) {
            // `ring_verdict`, because the band it is checked against is the
            // ring's horizon and the ring no longer decides. `reach_band` still
            // measures the ring, so the two must keep agreeing with each other.
            let aged = w.ring_verdict(&state, now - reach, &[3 * P]) == HostWriteVerdict::Aged;
            assert_eq!(
                reach_band(reach) == "hw_reach_in_ring",
                !aged,
                "reach {reach} bands as {} but the ring {} it",
                reach_band(reach),
                if aged { "refused" } else { "answered" }
            );
        }
    }

    /// A mapping-named write is resolved through the mapping's live page list, so
    /// a mapping re-pointed since must not be tested against its new pages.
    /// This is the one contract the swap to the page-keyed record changed, and
    /// it changed in the safe direction. The ring resolved a mapping-named write
    /// at *read* time and had to answer `Unresolvable` once the mapping moved —
    /// a correct refusal, but a refusal, and each one cost a re-gather. The
    /// record captures the page set at write time, so re-pointing the mapping
    /// afterwards cannot make the answer wrong: page 7 was written and still
    /// reads `Overlap`, page 3 was not and reads `Quiet`, whatever the mapping
    /// points at now.
    #[test]
    fn a_mapping_re_pointed_after_the_write_is_still_answered_by_the_pages_it_wrote() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        let mut state =
            crate::model::DeviceState::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_X86);
        state.map_surface(4);
        state.attach_mapping_internal(4, 0);
        let m = state.mappings.get_mut(&4).expect("just mapped");
        m.page_entries = vec![(7u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        let generation = m.map_generation;

        let mut w = HostWrites::default();
        let mark = w.epoch();
        w.note_mapping(4, generation, Some(&[7 * P]));
        assert_eq!(
            w.wrote_any_since(&state, mark, &[7 * P]),
            HostWriteVerdict::Overlap
        );
        assert_eq!(
            w.wrote_any_since(&state, mark, &[8 * P]),
            HostWriteVerdict::Quiet
        );

        // Re-point the mapping at a page the write never touched. The ring can
        // no longer reconstruct the write's page set; the record never needed to.
        let m = state.mappings.get_mut(&4).expect("still mapped");
        m.page_entries = vec![(8u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        m.map_generation = generation.wrapping_add(1);
        assert_eq!(
            w.wrote_any_since(&state, mark, &[7 * P]),
            HostWriteVerdict::Overlap,
            "the page this write landed in is still named by it"
        );
        assert_eq!(
            w.wrote_any_since(&state, mark, &[3 * P]),
            HostWriteVerdict::Quiet,
            "a page no write ever touched is quiet however the mapping has moved"
        );
    }

    /// A writer that cannot name every page of its mapping must not have the
    /// pages it *could* name mistaken for the whole write.
    #[test]
    fn a_mapping_write_that_cannot_name_all_its_pages_invalidates_everything_older() {
        let state =
            crate::model::DeviceState::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_X86);
        let mut w = HostWrites::default();
        let mark = w.epoch();
        w.note_mapping(4, 0, None);
        assert_eq!(
            w.wrote_any_since(&state, mark, &[3 * P]),
            HostWriteVerdict::Unnamed
        );
    }
}
