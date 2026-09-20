//! Guest pages the guest has taken back, and whether this device wrote to one
//! afterwards.
//!
//! # Why [`crate::runtime::node_guard`] is not enough
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
//! a finding.
//!
//! **The ordering claim this paragraph used to make is wrong and is worth not
//! repeating.** It said the guest submits the unmap, blocks on this device's
//! reply, and only then unwires — so that our stamp opened step 4. It does not
//! block: the submit and the unwire are adjacent, and measured, the guest
//! finishes unwiring before this device even reads the packet about nineteen
//! times in twenty. Nothing this device replies opens that window; the guest
//! opens it itself.
//!
//! So this watches the other end. A page released by the guest is a page this
//! device has been told to stop writing, whatever it later becomes, and a write
//! to one is a defect on its own terms — it does not need the page to have
//! become a page table for the answer to be "we wrote where we were told not
//! to". That the corrupting value must be a **zero** word for the guest's
//! assertion to fire is a property of the panic, not of this check.
//!
//! # The guard is armed at the page, not swept from a watch
//!
//! This was a `BTreeMap` of released pages, judged against the write census
//! once per drain tranche. That shape cost the watched population per tranche,
//! which is why it carried a capacity — and the capacity is what decided what
//! it could see. Every macos-13 boot logged `watching=4071 refused=58595`: the
//! instrument held 6 % of its own subject and had never once reported.
//!
//! Inverted, the release marker lives in the [`crate::runtime::host_writes`]
//! cell for that page — the same cell every writer that names its pages already
//! touches. Arming, disarming and detection are each one cell access, no sweep
//! exists, and the armed population is bounded only by the guest's own
//! releases. This module is what the readings *mean*; `host_writes` is where
//! they are kept.
//!
//! # Terminals, not a horizon
//!
//! A page stops being armed when **any** task maps it again — at which point
//! writing to it is legitimate, and keeping it would report ordinary work as a
//! defect — or when it reports. Neither is a duration chosen in advance.
//!
//! Note "any task". A guest page is guest-physical and more than one task can
//! map it — a shared IOSurface is exactly that — so a per-task watch arms a page
//! when task A unmaps it and then reports the perfectly legitimate write that
//! arrives through task B's live mapping. Armed globally, any task mapping the
//! page disarms it, and that whole class of false finding cannot arise.
//!
//! The residue, which no keying fixes: if A and B both map a page and only A
//! unmaps, the page is armed while B still holds it, and a write through B reads
//! as a finding. Counting live mappings per page would answer it, and it is not
//! done here because the count would have to be complete to be worth anything —
//! a page mapped by a route this guard does not see would make every later
//! answer for it wrong in the quiet direction. So the residue is stated, and the
//! discriminator for a specific finding is whether the page is still reachable
//! by any task at the time it fires.
//!
//! # What a finding means, and what it does not
//!
//! `released_write_after_release` means: between the guest releasing this page
//! and the write that reported it, this device wrote to it, and that write named
//! its pages. A write that names no pages cannot implicate a page or clear one;
//! those are counted separately rather than guessed at in either direction.
//!
//! It does **not** mean the write reached a page table. It means the ordering
//! this device relies on did not hold, which is the precondition for that.

/// Report the writes that landed on pages the guest had taken back.
///
/// Runs on the drain tranche. Unlike the sweep this replaced, the work here is
/// draining a queue that is empty on every tranche of every healthy boot; the
/// detection itself happened at the write.
pub fn sweep(state: &mut crate::model::DeviceState) {
    let writes = &mut state.host_writes;
    for hit in writes.take_released_writes() {
        crate::runtime::drain::note_store_route("released_write_after_release");
        if !crate::observe::first_sight("released_write_after_release", hit.gpa) {
            continue;
        }
        crate::observe::fail(format!(
            "released_pages reason=released_write_after_release gpa={:#x} \
             released_at={} wrote_at={} armed={} (this device wrote to a guest page after the \
             guest released it; the guest is entitled to have given that page to something \
             else, including its own page table)",
            hit.gpa,
            hit.released_at,
            hit.wrote_at,
            writes.armed_pages(),
        ));
    }
}

/// The guard's own levels, at most once per census interval.
///
/// `armed` is the population the guard is watching, and unlike the watch this
/// replaced it is the guest's number rather than a capacity. `dropped` non-zero
/// means findings arrived faster than the drain and the reported set is smaller
/// than the real one; `unnamed` counts the writes that could neither implicate
/// an armed page nor clear it.
///
/// **The cadence is enforced here and cannot be left to the call site.** This is
/// called from the drain tranche, beside [`sweep`], which genuinely wants to run
/// every tranche — so a levels line with no gate of its own inherits the tranche
/// rate. It did: a driven macos-13 window ran 363 tranches a second and this
/// emitted 8 297 lines over 25 s, **62 % of every line in the log**, while its
/// own doc said "on the census cadence".
///
/// The gate is [`crate::runtime::surface_cache::note_cache_levels`]'s, deliberately:
/// sharing the one-second interval is what lets a boot read this row-for-row
/// against `store_routes` and `drain_duty`.
pub fn note_levels(state: &crate::model::DeviceState) {
    static LAST_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    let writes = &state.host_writes;
    if writes.armed_pages() == 0
        && writes.dropped_released_writes() == 0
        && writes.unnamed_writes_while_armed() == 0
    {
        return;
    }
    if !claim_census_interval(&LAST_MS, crate::observe::elapsed_ms() as u64) {
        return;
    }
    crate::observe::off(format!(
        "released_pages_levels armed={} dropped={} unnamed={}",
        writes.armed_pages(),
        writes.dropped_released_writes(),
        writes.unnamed_writes_while_armed(),
    ));
}

// ---------------------------------------------------------------------------
// The read side
// ---------------------------------------------------------------------------
//
// # What the guards above leave unobserved
//
// Everything in this module so far watches a **write**: the release arms a
// page, the write census judges it where the write is recorded, and the drain
// reports whichever writes named an armed page. On the three panicking boots of
// the 2026-09-20 full-import round that reading was zero — no write this device
// recorded landed on a released page, and every panic-named page that a kernel
// slide could convert sat two to three frames outside the whole boot's write
// footprint.
//
// That leaves the other direction: this device **reading** a page the guest has
// taken back. A reader served from such a page is not reading a stale copy of
// its own data — the guest is entitled to have handed the page to something
// else, including its own page tables — so whatever it derives from those bytes
// is derived from somebody else's memory. Nothing here observed that, and it is
// the shape the remaining evidence points at.
//
// # The check, and what makes its zero an answer
//
// The marker for an armed page already exists in the write census cell
// ([`crate::runtime::host_writes::HostWrites::released_at`]), so the check is
// one lookup per page of a read that names its pages. What a named reader owes
// is therefore: name the pages it is about to read, ask each one, and report
// the ones the guest has released.
//
// Four counters make the zero readable rather than silent:
//
// * `read_guard_reads` — every named read this probe was called for while it is
//   on, and `read_guard_unarmed_reads` the subset that ran with nothing armed
//   to judge it against. Without these, a boot whose guest never released a
//   page and a boot whose wiring is dead read alike: both are zero everywhere
//   below.
// * `read_guard_checked_reads` / `read_guard_checked_pages` — how many named
//   reads were checked and over how many pages. A boot with hits is not a boot
//   with zero here, and a zero *here* would mean the probe never ran.
// * `read_guard_unnamed_reads` — a reader that could not name its pages. That
//   is not a clean sweep: it is a read this probe could not judge, and it is
//   reported instead of being counted as quiet.
// * `read_after_release` plus one `read_after_release_<reader>` per reader —
//   pages found armed, and by which reader.
//
// Each hit is also one `read_after_release` line on the always-on channel,
// latched per `(reader, page)`, carrying the release epoch and the reader's own
// description of the read. The line is the finding; the counters are what say
// how much of the boot the finding was measured against.

/// Whether the read-side probe observes anything this boot.
///
/// Affirmative spelling only, unlike [`crate::runtime::node_guard::enabled`]:
/// the check is a page walk per named read, and a boot that is not the
/// read-side round must not pay it or perturb the race it exists to watch.
/// `unset`, `off` and an unrecognized value all mean "do not observe".
pub fn reads_guarded() -> bool {
    // The tests below are the only caller that may answer this without the
    // environment, because the crate reads the switch once per process and a
    // fixture cannot un-cache that. This changes nothing a boot can reach: it
    // is compiled out of every arm but `cfg(test)`.
    #[cfg(test)]
    if FORCED_ON.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        crate::config::switch(crate::config::READ_GUARD) == crate::config::Switch::On
    })
}

#[cfg(test)]
static FORCED_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The probe on, for the duration of one test.
///
/// The read of the switch is cached per process, so a fixture that set the
/// environment and put it back could not turn the probe on for its own window.
/// The override is `cfg(test)`, so no arm can reach it — and it is published
/// `pub(crate)` rather than kept to this module's own tests, because the
/// entries *outside* this module ([`crate::backend::provider_render`]'s
/// sampled bind is one) have to prove their own call reaches the probe, and a
/// test that could only ask the switch would prove nothing about the wiring.
#[cfg(test)]
pub(crate) struct ForcedOn;

#[cfg(test)]
impl ForcedOn {
    pub(crate) fn arm() -> Self {
        FORCED_ON.store(true, std::sync::atomic::Ordering::Relaxed);
        Self
    }
}

#[cfg(test)]
impl Drop for ForcedOn {
    fn drop(&mut self) {
        FORCED_ON.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Census route naming one reader's hits, derived from the reader's own settle
/// slug so the probe cannot grow a second vocabulary for the same sites.
///
/// `settle_buffer_guest_read` becomes `read_after_release_buffer_guest_read`.
/// Derived once per site and leaked, because [`crate::runtime::drain::note_store_route`]
/// takes a `&'static str` and a reader that has to allocate its own route name
/// would pay that on every checked read.
pub fn site_route(site: crate::runtime::render_writeback::SettleSite) -> &'static str {
    use crate::runtime::render_writeback::SettleSite;
    static ROUTES: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    let routes = ROUTES.get_or_init(|| {
        SettleSite::ALL
            .iter()
            .map(|site| {
                let slug = site.route();
                let reader = slug.strip_prefix("settle_").unwrap_or(slug);
                Box::leak(format!("read_after_release_{reader}").into_boxed_str()) as &'static str
            })
            .collect()
    });
    let index = SettleSite::ALL
        .iter()
        .position(|candidate| *candidate == site)
        .unwrap_or(0);
    routes[index]
}

/// Route naming the bind-time check: a draw-time buffer window this device is
/// about to serve, whose pages are resolved from the window rather than from a
/// settle site.
pub const BIND_ROUTE: &str = "read_after_release_buffer_bind";

/// Route naming the provider rail's own bind-time check: a sampled texture
/// whose texels are the guest's pages, resolved by
/// [`crate::backend::provider_render`] rather than by the draw path's buffer
/// resolution.
///
/// A second constant and not a second spelling of [`BIND_ROUTE`]: the two
/// entries name two different reads of the same boot — a buffer this device
/// binds at draw time, and a texture the canonical rail gathers out of the
/// guest's own pages — and a census that could not tell them apart could not
/// say which of the two families a hit came from.
pub const SAMPLED_BIND_ROUTE: &str = "read_after_release_sampled_bind";

/// The provider rail's sampled-bind entry, counted on its own: every call the
/// entry made with the probe on.
///
/// The shared [`note_read`] counters say how much of a boot the whole probe
/// judged; they do not say which *entry* the reads came from, and for the
/// sampled family that is the whole question — 148 borrowed windows and 48
/// repacked rows across one round's two 240 s arms are the population the entry
/// exists to watch, and a reader of the log has to be able to size it. Charged
/// before the armed-set test, so it counts a read this entry saw and could not
/// judge as well as one it judged.
pub const SAMPLED_BIND_READS: &str = "read_guard_sampled_bind_reads";

/// The sampled-bind entry's reads that reached its page resolve — the analogue
/// of [`note_read`]'s `read_guard_checked_reads`, for this entry alone.
///
/// Charged where the resolve runs, which is the same condition the shared
/// counter's `checked` arm is under (the probe is on and the armed set is not
/// empty), so the two cannot disagree about which reads were checkable.
pub const SAMPLED_BIND_CHECKED_READS: &str = "read_guard_sampled_bind_checked_reads";

/// The pages the sampled-bind entry's own resolve named, summed the way
/// `read_guard_checked_pages` is.
///
/// A read whose pages this entry could not name (a source with no run list, a
/// packed alias) contributes nothing here and is counted by the shared
/// `read_guard_unnamed_reads` instead: the two are different readings and must
/// not be added into one number.
pub const SAMPLED_BIND_CHECKED_PAGES: &str = "read_guard_sampled_bind_checked_pages";

/// Check one named read's pages against the released set.
///
/// `pages` is the caller's own resolve of everything it is about to read, and
/// runs **only** when this probe is on and something is armed — the whole point
/// of the shape, because that resolve is a page-table walk and the common answer
/// on every other boot is that nothing is armed at all. `None` from it means the
/// caller could not name its window, which is counted as undecided rather than
/// as quiet; a short list would license the pages it omitted.
///
/// `describe` runs only on a hit, so a reader may put its task, span and
/// resource into the line without paying for the string on every read.
pub fn note_read(
    writes: &crate::runtime::host_writes::HostWrites,
    route: &'static str,
    pages: impl FnOnce() -> Option<Vec<u64>>,
    describe: impl Fn() -> String,
) {
    if !reads_guarded() {
        return;
    }
    // Counted before the armed-set test, because the two answers a reader has
    // to tell apart are "the probe saw reads and nothing was armed" and "the
    // probe saw no reads at all". The first is a reading about the boot's
    // unmap traffic; the second means the wiring is dead. One atomic add on a
    // path that is already a hash lookup when the set is empty.
    crate::runtime::drain::note_store_route("read_guard_reads");
    if writes.armed_pages() == 0 {
        crate::runtime::drain::note_store_route("read_guard_unarmed_reads");
        return;
    }
    let Some(pages) = pages() else {
        crate::runtime::drain::note_store_route("read_guard_unnamed_reads");
        return;
    };
    if pages.is_empty() {
        crate::runtime::drain::note_store_route("read_guard_unnamed_reads");
        return;
    }
    crate::runtime::drain::note_store_route("read_guard_checked_reads");
    crate::runtime::drain::note_store_route_n("read_guard_checked_pages", pages.len() as u64);
    let mut reported: Vec<u64> = Vec::new();
    let mut hits = 0u64;
    for &gpa in &pages {
        let Some(released_at) = writes.released_at(gpa) else {
            continue;
        };
        if reported.contains(&gpa) {
            continue;
        }
        reported.push(gpa);
        hits += 1;
        crate::runtime::drain::note_store_route(route);
        // The latch is per `(reader, page)`, so one page read by two readers is
        // two lines and one page read twice by the same reader is one.
        if !crate::observe::first_sight(route, gpa) {
            continue;
        }
        crate::observe::fail(format!(
            "{route} reason=read_after_release gpa={gpa:#x} released_at={released_at} \
             armed={} pages={} ({}; this device read a guest page the guest had already taken \
             back and has not mapped again, so the bytes it read belong to whatever the guest \
             put there next — the read side of the ordering the release guards watch on the \
             write side. This probe observes and never decides: the guest-visible behavior of \
             this boot is unchanged)",
            writes.armed_pages(),
            pages.len(),
            describe(),
        ));
    }
    crate::runtime::drain::note_store_route_n("read_after_release", hits);
}

/// The probe's own positive control, run once when the guard is on.
///
/// The zero this probe reports is only worth something if the query it reports
/// it with answers both ways on this build, and the write-side guards learned
/// that lesson the expensive way: a rate gate written inline against the clock
/// could only be checked by a boot. This runs against a detached
/// [`crate::runtime::host_writes::HostWrites`] — it arms a page nothing else
/// can see, asks for it, disarms it and asks again — so it proves the lookup
/// without touching the device's own armed population or perturbing a single
/// judgment the boot makes.
pub fn note_selftest() {
    use crate::runtime::host_writes::HostWrites;

    if !reads_guarded() {
        return;
    }
    let page = 0x1000u64;
    let mut probe = HostWrites::new(crate::model::PAGE_SHIFT_X86);
    probe.release_page(page);
    let armed = probe.released_at(page);
    let disarmed = if probe.released_at(page + 0x1000).is_none() {
        "miss"
    } else {
        "HIT-WITHOUT-A-RELEASE"
    };
    let cleared = {
        probe.remap_page(page);
        probe.released_at(page).is_none()
    };
    crate::observe::off(format!(
        "read_guard_selftest armed_query={} on_released_elsewhere={} after_remap={} \
         (armed_query must be hit and the other two miss; a boot in which this line is absent \
         ran no read-side probe)",
        match armed {
            Some(_) => "hit",
            None => "MISS-WITHOUT-A-RELEASE",
        },
        disarmed,
        if cleared { "miss" } else { "HIT-AFTER-REMAP" },
    ));
}

/// The census interval every levels line in this device shares.
///
/// One second, so `released_pages_levels`, `cache_levels`, `store_routes` and
/// `drain_duty` all describe the same window and can be read as one row.
const CENSUS_INTERVAL_MS: u64 = 1_000;

/// Atomically claim the next census interval, or refuse.
///
/// Split out and given the clock as an argument for the same reason
/// `drain::claim_display_vbl` is: a rate gate written inline against
/// `elapsed_ms()` can only be checked by a boot, and this one was wrong for as
/// long as it was inline. Here it is a pure function of `(last, now)` and the
/// tests below pin both edges.
///
/// The claimed timestamp is set to `now` rather than advanced by one interval —
/// the opposite of the VBL grid, and deliberately. A VBL is a cadence the guest
/// latches onto, so its phase must not drift; this is a sample of a level, where
/// landing exactly on a grid buys nothing and back-dating would let a burst of
/// tranches after a long stall each emit a line.
fn claim_census_interval(last_ms: &std::sync::atomic::AtomicU64, now_ms: u64) -> bool {
    use std::sync::atomic::Ordering;
    let last = last_ms.load(Ordering::Relaxed);
    if now_ms.saturating_sub(last) < CENSUS_INTERVAL_MS {
        return false;
    }
    // Losing the race only costs a skipped interval, never a double line.
    last_ms
        .compare_exchange(last, now_ms, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
}

#[cfg(test)]
mod tests {
    use crate::runtime::host_writes::HostWrites;
    use crate::runtime::render_writeback::SettleSite;

    use super::*;

    const P: u64 = 4096;

    fn route_count(route: &str) -> u64 {
        crate::runtime::drain::store_route_count(route)
    }

    /// A reader of a page the guest has released is the finding: one line naming
    /// the reader, the page and the release epoch, the reader's own route
    /// counted, and the checked counters moved so the zero case stays readable.
    #[test]
    fn a_read_of_a_released_page_is_reported_against_the_reader() {
        let _on = ForcedOn::arm();
        let cap = crate::observe::FailCapture::start();
        let mut w = HostWrites::default();
        w.release_page(9 * P);
        let checked_before = route_count("read_guard_checked_reads");
        let pages_before = route_count("read_guard_checked_pages");
        let total_before = route_count("read_after_release");
        let reader_before = route_count(BIND_ROUTE);

        note_read(
            &w,
            BIND_ROUTE,
            || Some(vec![3 * P, 9 * P]),
            || "reader=fixture task=1".to_string(),
        );

        let line = cap.one(BIND_ROUTE);
        assert!(line.contains("reason=read_after_release"), "{line}");
        assert!(line.contains(&format!("gpa={:#x}", 9 * P)), "{line}");
        assert!(!line.contains(&format!("gpa={:#x}", 3 * P)), "{line}");
        assert!(line.contains("reader=fixture task=1"), "{line}");
        assert_eq!(route_count(BIND_ROUTE) - reader_before, 1);
        assert_eq!(route_count("read_after_release") - total_before, 1);
        assert_eq!(route_count("read_guard_checked_reads") - checked_before, 1);
        assert_eq!(route_count("read_guard_checked_pages") - pages_before, 2);
    }

    /// The negative control: a read of pages the guest still holds is checked
    /// and quiet, and the check is counted. Without this a zero reader could not
    /// tell "nothing read a released page" from "nothing was read".
    #[test]
    fn a_read_of_a_page_the_guest_still_holds_is_checked_and_quiet() {
        let _on = ForcedOn::arm();
        let cap = crate::observe::FailCapture::start();
        let mut w = HostWrites::default();
        w.release_page(9 * P);
        let checked_before = route_count("read_guard_checked_reads");
        let reader_before = route_count(BIND_ROUTE);

        note_read(
            &w,
            BIND_ROUTE,
            || Some(vec![4 * P]),
            || "unused".to_string(),
        );

        assert!(cap.lines().is_empty(), "{:?}", cap.lines());
        assert_eq!(route_count(BIND_ROUTE), reader_before);
        assert_eq!(route_count("read_guard_checked_reads") - checked_before, 1);
    }

    /// A reader that cannot name its window is undecided, not quiet — the same
    /// direction the write census refuses in. An empty window is the same
    /// answer: a read over no page this probe could judge.
    #[test]
    fn a_reader_that_cannot_name_its_pages_is_counted_and_not_called_quiet() {
        let _on = ForcedOn::arm();
        let _cap = crate::observe::FailCapture::start();
        let mut w = HostWrites::default();
        w.release_page(9 * P);
        let unnamed_before = route_count("read_guard_unnamed_reads");

        note_read(&w, BIND_ROUTE, || None, || "unused".to_string());
        note_read(&w, BIND_ROUTE, || Some(Vec::new()), || "unused".to_string());

        assert_eq!(route_count("read_guard_unnamed_reads") - unnamed_before, 2);
    }

    /// The probe is off unless the switch is spelled on, and off means **off**:
    /// the reader's own resolve does not run, so a boot that is not watching
    /// pays neither the walk nor the check.
    #[test]
    fn the_probe_runs_nothing_when_it_is_off() {
        let cap = crate::observe::FailCapture::start();
        let mut w = HostWrites::default();
        w.release_page(9 * P);
        let checked_before = route_count("read_guard_checked_reads");
        let mut resolves = 0u32;

        note_read(
            &w,
            BIND_ROUTE,
            || {
                resolves += 1;
                Some(vec![9 * P])
            },
            || "unused".to_string(),
        );

        assert_eq!(resolves, 0, "the resolve must not run with the probe off");
        assert_eq!(route_count("read_guard_checked_reads"), checked_before);
        assert!(cap.lines().is_empty(), "{:?}", cap.lines());
    }

    /// With nothing armed there is nothing a read could hit, and the resolve is
    /// not paid for either — but the call is still counted, so the two zero
    /// readings ("saw reads, nothing armed" and "saw nothing") stay apart.
    #[test]
    fn the_probe_pays_nothing_while_the_armed_set_is_empty() {
        let _on = ForcedOn::arm();
        let w = HostWrites::default();
        let mut resolves = 0u32;
        let reads_before = route_count("read_guard_reads");
        let unarmed_before = route_count("read_guard_unarmed_reads");
        let checked_before = route_count("read_guard_checked_reads");
        note_read(
            &w,
            BIND_ROUTE,
            || {
                resolves += 1;
                Some(vec![9 * P])
            },
            || "unused".to_string(),
        );
        assert_eq!(resolves, 0);
        assert_eq!(route_count("read_guard_reads") - reads_before, 1);
        assert_eq!(route_count("read_guard_unarmed_reads") - unarmed_before, 1);
        assert_eq!(route_count("read_guard_checked_reads"), checked_before);
        assert_eq!(w.armed_pages(), 0);
    }

    /// The route a reader's hits are counted under is derived from that
    /// reader's own settle slug, so the probe cannot grow a second name for a
    /// site — and two readers cannot share one.
    #[test]
    fn the_probes_routes_are_derived_from_the_readers_own_slugs() {
        assert_eq!(
            site_route(SettleSite::BufferGuestRead),
            "read_after_release_buffer_guest_read"
        );
        let routes: Vec<&'static str> = SettleSite::ALL.iter().copied().map(site_route).collect();
        assert_eq!(routes.len(), SettleSite::ALL.len());
        let unique: std::collections::HashSet<&'static str> = routes.iter().copied().collect();
        assert_eq!(unique.len(), routes.len(), "two readers share one route");
        for route in routes {
            assert!(route.starts_with("read_after_release_"), "{route}");
        }
        // The two bind-time entries name two different reads — a draw's buffer
        // window and the provider rail's sampled gather — so a census line can
        // say which family a hit came from. One name for both would make the
        // finding unassignable, which is the whole reason each entry carries
        // its own route.
        assert_eq!(BIND_ROUTE, "read_after_release_buffer_bind");
        assert_eq!(SAMPLED_BIND_ROUTE, "read_after_release_sampled_bind");
        assert_ne!(BIND_ROUTE, SAMPLED_BIND_ROUTE);
        for route in [BIND_ROUTE, SAMPLED_BIND_ROUTE] {
            assert!(route.starts_with("read_after_release_"), "{route}");
            assert!(
                !unique.contains(route),
                "a bind route collides with a settle-derived route: {route}"
            );
        }
        assert!(
            crate::config::ALL.contains(&crate::config::READ_GUARD),
            "the switch must be on the boot line"
        );
    }

    /// The probe's own control answers both ways on this build, and says so on
    /// a line a boot can be read for.
    #[test]
    fn the_selftest_answers_both_ways_and_says_so() {
        let _on = ForcedOn::arm();
        let cap = crate::observe::FailCapture::start();
        note_selftest();
        // An `off` line, so the slug is the second token and `one` (which
        // matches the first) would find nothing.
        let lines = cap.lines();
        let line = lines
            .iter()
            .find(|line| line.contains("read_guard_selftest"))
            .unwrap_or_else(|| panic!("no read_guard_selftest line in {lines:?}"))
            .clone();
        assert!(line.contains("armed_query=hit"), "{line}");
        assert!(line.contains("on_released_elsewhere=miss"), "{line}");
        assert!(line.contains("after_remap=miss"), "{line}");
    }

    /// A read of a page the guest released reports once per reader and page,
    /// however many times the same reader reads it.
    #[test]
    fn one_page_read_twice_reports_once_and_disarms_nothing() {
        let _on = ForcedOn::arm();
        let cap = crate::observe::FailCapture::start();
        let mut w = HostWrites::default();
        w.release_page(9 * P);
        note_read(&w, BIND_ROUTE, || Some(vec![9 * P]), || "first".to_string());
        note_read(
            &w,
            BIND_ROUTE,
            || Some(vec![9 * P]),
            || "second".to_string(),
        );
        assert_eq!(cap.lines().len(), 1, "{:?}", cap.lines());
        assert_eq!(
            w.armed_pages(),
            1,
            "a read is not a write: the page stays armed for the write guard"
        );
        assert!(
            w.take_released_writes().is_empty(),
            "the read-side probe must not manufacture a write finding"
        );
    }

    /// A released page nobody writes to reports nothing and stays armed — it is
    /// waiting for a write that may still come.
    #[test]
    fn a_released_page_nobody_wrote_to_stays_armed_and_quiet() {
        let mut w = HostWrites::default();
        w.release_page(9 * P);
        w.note_pages(vec![3 * P, 4 * P]);
        assert!(w.take_released_writes().is_empty());
        assert_eq!(w.armed_pages(), 1);
    }

    /// A write after the release is the finding, it carries both epochs, and it
    /// is reported exactly once.
    #[test]
    fn a_write_after_the_release_is_reported_once() {
        let mut w = HostWrites::default();
        w.note_pages(vec![P]);
        let released_at = w.epoch();
        w.release_page(9 * P);
        w.note_pages(vec![9 * P]);

        let found = w.take_released_writes();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].gpa, 9 * P);
        assert_eq!(found[0].released_at, released_at);
        assert_eq!(found[0].wrote_at, w.epoch());
        assert_eq!(
            w.armed_pages(),
            0,
            "a page that reported is not armed again"
        );

        w.note_pages(vec![9 * P]);
        assert!(
            w.take_released_writes().is_empty(),
            "one late write is one finding, not one per write for the rest of the boot"
        );
    }

    /// A write *before* the release says nothing — that is ordinary work on a
    /// page the guest still wanted us in.
    #[test]
    fn a_write_before_the_release_is_not_a_finding() {
        let mut w = HostWrites::default();
        w.note_pages(vec![9 * P]);
        w.release_page(9 * P);
        assert!(w.take_released_writes().is_empty());
        assert_eq!(w.armed_pages(), 1, "still armed, still waiting");
    }

    /// A page the guest maps again leaves the guard, so the writes that follow
    /// are ordinary work and not findings. Without this every recycled page
    /// would report.
    #[test]
    fn a_remapped_page_leaves_the_guard() {
        let mut w = HostWrites::default();
        w.release_page(9 * P);
        w.remap_page(9 * P);
        assert_eq!(w.armed_pages(), 0);
        w.note_pages(vec![9 * P]);
        assert!(w.take_released_writes().is_empty());
    }

    /// A page released by one task and mapped by **another** is disarmed.
    ///
    /// This is the whole reason the guard is not keyed by task. A shared surface
    /// is mapped by more than one task, so keying by task would arm the page on
    /// the first unmap and then report every legitimate write arriving through
    /// the other task's live mapping — a finding per shared page per boot, all
    /// of them wrong, in an instrument whose only value is that a hit is a
    /// proof. The guard is page-keyed, so there is no task to key by.
    #[test]
    fn a_page_mapped_again_by_a_different_task_is_disarmed() {
        let mut w = HostWrites::default();
        w.release_page(9 * P);
        w.remap_page(9 * P);
        w.note_pages(vec![9 * P]);
        assert!(
            w.take_released_writes().is_empty(),
            "a write through the other task's live mapping is ordinary work"
        );
    }

    /// Releasing a page twice keeps the first epoch, so a write that already
    /// happened is not forgiven by the second release.
    #[test]
    fn a_second_release_does_not_forgive_a_write_that_already_landed() {
        let mut w = HostWrites::default();
        w.note_pages(vec![P]);
        let first = w.epoch();
        w.release_page(9 * P);
        w.note_pages(vec![7 * P]);
        w.release_page(9 * P);
        w.note_pages(vec![9 * P]);
        let found = w.take_released_writes();
        assert_eq!(found.len(), 1);
        assert_eq!(
            found[0].released_at, first,
            "the finding is measured from the first release, not the second"
        );
    }

    /// A write that named no pages cannot judge an armed one, in either
    /// direction, and says so rather than reading as a clean sheet.
    #[test]
    fn an_unnamed_write_is_counted_and_is_not_a_finding() {
        let mut w = HostWrites::default();
        w.note_unknown();
        assert_eq!(
            w.unnamed_writes_while_armed(),
            0,
            "nothing armed, nothing to be undecided about"
        );
        w.release_page(9 * P);
        w.note_unknown();
        assert!(w.take_released_writes().is_empty());
        assert_eq!(w.unnamed_writes_while_armed(), 1);
        assert_eq!(
            w.armed_pages(),
            1,
            "an unnamed write neither clears nor arms"
        );
    }

    /// **The population is the guest's, not a capacity.** The watch this
    /// replaced held 4096 pages and every macos-13 boot refused 58 595 of the
    /// 62 666 the guest released — 93 % of the instrument's own subject, unseen.
    /// A release of that size is now armed in full and each page still answers.
    #[test]
    fn sixty_thousand_released_pages_are_all_armed_and_all_answer() {
        const N: u64 = 62_666;
        let mut w = HostWrites::default();
        w.note_pages(vec![0]);
        for page in 1..=N {
            w.release_page(page * P);
        }
        assert_eq!(w.armed_pages(), N);
        // The last page released is well past any capacity the watch had.
        w.note_pages(vec![N * P]);
        let found = w.take_released_writes();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].gpa, N * P);
        assert_eq!(w.armed_pages(), N - 1);
    }

    /// A page armed before this device has ever written is still armed. Epoch 0
    /// means "never written", so an arm that stored it would be indistinguishable
    /// from an unarmed cell and the guard would be blind for the whole first
    /// write of a boot.
    #[test]
    fn a_page_released_before_the_first_write_is_still_armed() {
        let mut w = HostWrites::default();
        assert_eq!(w.epoch(), 0, "nothing written yet");
        w.release_page(9 * P);
        assert_eq!(w.armed_pages(), 1);
        w.note_pages(vec![9 * P]);
        assert_eq!(w.take_released_writes().len(), 1);
    }

    /// A whole-chunk write is the one path that does not touch cells one by one,
    /// and it must still find an armed page inside the chunk it covers.
    #[test]
    fn a_whole_chunk_write_finds_an_armed_page_inside_it() {
        let pages: std::sync::Arc<[u64]> =
            (0..512u64).map(|page| page * P).collect::<Vec<_>>().into();
        let footprint = crate::runtime::guest_ram::GuestPageFootprint::new(pages, P)
            .expect("one contiguous allocation");
        let mut w = HostWrites::default();
        w.release_page(300 * P);
        w.note_footprint(&footprint);
        let found = w.take_released_writes();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].gpa, 300 * P);
    }

    /// The report queue bounds the alarms, not the guard. What it cannot hold is
    /// counted, and every page it did not report is still disarmed exactly once.
    #[test]
    fn a_flood_of_findings_is_counted_rather_than_lost_silently() {
        let mut w = HostWrites::default();
        w.note_pages(vec![0]);
        let pages: Vec<u64> = (1..=200u64).map(|page| page * P).collect();
        for &gpa in &pages {
            w.release_page(gpa);
        }
        w.note_pages(pages);
        let found = w.take_released_writes();
        assert_eq!(found.len() as u64 + w.dropped_released_writes(), 200);
        assert!(
            w.dropped_released_writes() > 0,
            "200 findings, a queue of 64"
        );
        assert_eq!(
            w.armed_pages(),
            0,
            "a dropped report still disarms its page"
        );
    }

    /// Arm64's wider pages occupy one cell, so a release and a write of the same
    /// guest page cannot land in different cells.
    #[test]
    fn arm64_pages_arm_and_report_at_arm64_geometry() {
        const ARM_PAGE: u64 = 1 << crate::model::PAGE_SHIFT_ARM64E;
        let mut w = HostWrites::new(crate::model::PAGE_SHIFT_ARM64E);
        w.release_page(3 * ARM_PAGE);
        // An address inside the same 16 KiB page, not at its base.
        w.note_pages(vec![3 * ARM_PAGE + 4096]);
        let found = w.take_released_writes();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].gpa, 3 * ARM_PAGE);
    }

    /// The gate that makes this a census line and not a per-tranche one.
    ///
    /// A driven macos-13 window runs ~363 drain tranches a second and
    /// `note_levels` is called from every one of them, so without a claim here
    /// the level is emitted 363 times a second. This asserts the tranche rate
    /// cannot get through: 363 calls spread across one second yield one line.
    #[test]
    fn a_seconds_worth_of_tranches_claims_one_census_interval() {
        let last = std::sync::atomic::AtomicU64::new(0);
        // Start at a nonzero clock so the first call is not a special case of
        // the zero initialiser.
        let base = 5_000;
        assert!(claim_census_interval(&last, base), "the first sample emits");

        // Up to 362, not 363: the 363rd tranche lands exactly on `base + 1000`,
        // which is a full interval later and is *supposed* to claim. It is the
        // next assertion, not part of this one.
        let claims = (1..363)
            .filter(|i| claim_census_interval(&last, base + (1000 * i) / 363))
            .count();
        assert_eq!(claims, 0, "a tranche inside the interval must not emit");

        assert!(
            claim_census_interval(&last, base + CENSUS_INTERVAL_MS),
            "the tranche that reaches the next interval emits"
        );
    }

    /// A stall does not bank intervals it slept through.
    ///
    /// The claim moves to `now`, not forward by one interval, so the tranches
    /// that arrive in a burst after a long drain stall produce one line between
    /// them rather than one per interval the stall covered.
    #[test]
    fn a_long_stall_does_not_release_a_burst_of_lines() {
        let last = std::sync::atomic::AtomicU64::new(0);
        assert!(claim_census_interval(&last, 1_000));
        // Ten intervals pass inside one tranche, then the burst arrives.
        assert!(claim_census_interval(&last, 11_000));
        let banked = (1..=10)
            .filter(|i| claim_census_interval(&last, 11_000 + i))
            .count();
        assert_eq!(banked, 0, "the stall must not bank a line per interval");
    }
}
