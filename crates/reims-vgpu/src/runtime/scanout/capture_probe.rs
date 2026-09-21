//! The present capture's own step census, behind
//! [`crate::config::CAPTURE_PROBE`].
//!
//! # Why this exists beside `present_capture FAIL`
//!
//! `capture_present_frame` already says when it could not find a frame: the
//! failure channel gets `present_capture FAIL … reason=no_resident_content`
//! and the drain adds `present capture fail … keep_prior=N`. Both name the
//! *outcome* — the console keeps its prior retain — and neither names the
//! *step*, so a boot that reads like "the window is frozen because the capture
//! keeps missing" cannot be told from one where the capture misses once at the
//! first present of a surface the guest has not drawn into yet.
//!
//! Those two are the same line and different defects. The first is a
//! present-path fault; the second is the guest presenting a surface before
//! anything rendered into it, which is a real event on this device (the
//! compositor's first swap of a fresh buffer) and one a display is entitled to
//! answer by holding the previous frame.
//!
//! # What is counted
//!
//! One attempt is one call to `capture_present_frame` that reached the
//! resident-direct source. It is served by exactly one of
//! [`Source::HostCache`] and [`Source::Resident`], or it fails, and a failure
//! is classified by the first step that refused. The classes are deliberately
//! the four states [`crate::config::CAPTURE_PROBE`]'s doc lists, because each
//! has a different repair.
//!
//! # The other half of the pipeline, on the same switch
//!
//! The capture is one of two places that ask the registry for a resident, and
//! they ask the *same* four questions: the capture wants an image to hand the
//! console, the window **publish** wants one to hand the host window. The
//! publish's answers are [`PresentRefusal`], reported here rather than beside
//! the window because the reading that matters is the two halves together — a
//! boot whose captures are served and whose publishes are refused is a
//! different defect from one where neither finds a resident.
//!
//! Until this existed the publish's four answers reached the drain's route
//! channel as one word per class and stopped there: `winpub_no_resident` is
//! what a boot at `direct_frac=0.00` reported for its whole run, and that one
//! name covers both "nothing names this surface" and "the registry holds it
//! under another key" — the two states this probe's own doc says have different
//! repairs. [`PresentCounts::key_*`] separates them, on the same principle as
//! [`CaptureRefusal::KeyGeneration`] does one layer down.
//!
//! # Cost, and the shape of the output
//!
//! Off: [`enabled`] is one `OnceLock` load and nothing below runs — no counter
//! is touched, no string is built. On: one relaxed atomic addition per attempt,
//! and on a failure one registry report (already under a lock the capture holds)
//! plus, at most [`MAX_DETAILS`] lines per boot, deduplicated by signature.
//!
//! The summary is emitted at power-of-two attempt counts rather than per
//! attempt, the same rule `capture_sampling` uses: the ratio is readable from
//! the first attempt onward and a boot cannot turn one hot present into a flood.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use super::CaptureRefusal;

/// Distinct failure signatures that get a detail line in one boot.
///
/// A signature is (presented mid, the two mapping fields, the failing step), so
/// eight covers every mid a boot presents before it starts repeating one. Past
/// that the summary line still carries the counts; only the registry dump is
/// dropped, which is the expensive half.
pub const MAX_DETAILS: usize = 8;

/// Distinct present-publish signatures that get a detail line in one boot.
///
/// A budget of its own rather than a share of [`MAX_DETAILS`], so the two halves
/// cannot spend each other's: a boot whose captures miss six ways would
/// otherwise leave two lines for the publishes, which is where the question
/// that made this probe necessary is asked.
pub const PRESENT_MAX_DETAILS: usize = 8;

/// Why a window publish could not hand the host window a resident.
///
/// The publish's own four states, in the same order and with the same repairs
/// as [`CaptureRefusal`]'s first four: the answers
/// [`crate::backend::vulkan::engine::pools::slot_present_decline`] can give.
/// They used to collapse into one `bool` at the publish and reach the drain as
/// one more route, which is why a boot reading `direct_frac=0.00` in every
/// window named no cause at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentRefusal {
    /// No resident under the key the publish asked under. Which of the two
    /// states that is — nothing names the surface, or it is registered under a
    /// different key — is the divergence counted beside this one.
    NoResident,
    /// The image is registered and nothing has vouched for its pixels yet.
    ContentNotReady,
    /// The image's texels are not in the byte order the scanout blit reads.
    ScanoutOrder,
    /// The registered image is not at the extent being presented.
    Geometry,
}

impl PresentRefusal {
    /// The word this refusal contributes to `present_refused=`.
    ///
    /// Deliberately the same four words the capture side uses for the same four
    /// states, so one reader can put the two halves side by side.
    pub fn slug(self) -> &'static str {
        match self {
            Self::NoResident => "no_resident",
            Self::ContentNotReady => "content_not_ready",
            Self::ScanoutOrder => "scanout_order",
            Self::Geometry => "geometry",
        }
    }

    /// Every refusal, in the order the probe's summary line prints them.
    pub const ALL: [PresentRefusal; 4] = [
        Self::NoResident,
        Self::ContentNotReady,
        Self::ScanoutOrder,
        Self::Geometry,
    ];
}

/// Which source served a capture attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The host render cache held the frame (an encode or a clear wrote it).
    HostCache,
    /// The host cache had no bytes but named a cession to the resident, and the
    /// rail's readback served the frame.
    Resident,
    /// Neither source produced the frame; the console keeps its prior retain.
    None,
}

impl Source {
    /// The step this source is reported as. One name per source, so a reader
    /// never has to combine two lines to see which vein served a present.
    fn slug(self) -> &'static str {
        match self {
            Self::HostCache => "host_cache",
            Self::Resident => "resident",
            Self::None => "fail",
        }
    }
}

/// The refusal's slot in [`Counts::steps`].
///
/// Derived from the shared vocabulary rather than written out again, so a
/// refusal added to [`CaptureRefusal`] is a compile error here until it is
/// counted — which is the difference between a census that covers the
/// vocabulary and one that silently under-reports it.
fn step_index(refusal: CaptureRefusal) -> usize {
    match refusal {
        CaptureRefusal::NoTarget => 0,
        CaptureRefusal::KeyGeneration => 1,
        CaptureRefusal::KeyGeometry => 2,
        CaptureRefusal::KeyOther => 3,
        CaptureRefusal::ContentNotReady => 4,
        CaptureRefusal::ReadbackDeclined => 5,
        CaptureRefusal::NoRegistry => 6,
    }
}

/// One boot's counts. All relaxed: this is a census, and every one of these is
/// read long after the attempt it counts.
#[derive(Default)]
struct Counts {
    attempts: AtomicU64,
    host_cache: AtomicU64,
    host_cache_ceded: AtomicU64,
    resident: AtomicU64,
    fail: AtomicU64,
    steps: [AtomicU64; CaptureRefusal::ALL.len()],
}

static COUNTS: Counts = Counts {
    attempts: AtomicU64::new(0),
    host_cache: AtomicU64::new(0),
    host_cache_ceded: AtomicU64::new(0),
    resident: AtomicU64::new(0),
    fail: AtomicU64::new(0),
    steps: [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ],
};

/// Signatures that already have a detail line, and how often each has failed.
static DETAILED: Mutex<Vec<((&'static str, String), u64)>> = Mutex::new(Vec::new());

/// One boot's window-publish counts, on the same switch and reported on the
/// same summary line as the capture's.
///
/// The class counters are not the whole reading and are not meant to be: the
/// publish's returned slug already reaches `store_routes` as `winpub_<class>`
/// with the probe off. What only this can report is the **key** the refusal
/// happened under — the divergence behind `no_resident`, which is the class
/// every refused publish on this device has landed in.
struct PresentCounts {
    refusals: AtomicU64,
    classes: [AtomicU64; PresentRefusal::ALL.len()],
    /// Nothing in the registry names this surface at all.
    key_absent: AtomicU64,
    /// The surface is there and the key moved (the guest re-mapped it).
    key_generation: AtomicU64,
    /// The surface is there at another extent.
    key_geometry: AtomicU64,
    /// The surface and the extent agree and some other field differs.
    key_other: AtomicU64,
    /// The id is registered in another namespace, so it is not this object.
    key_namespace: AtomicU64,
}

static PRESENT_COUNTS: PresentCounts = PresentCounts {
    refusals: AtomicU64::new(0),
    classes: [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ],
    key_absent: AtomicU64::new(0),
    key_generation: AtomicU64::new(0),
    key_geometry: AtomicU64::new(0),
    key_other: AtomicU64::new(0),
    key_namespace: AtomicU64::new(0),
};

/// Signatures that already have a present-publish detail line.
static PRESENT_DETAILED: Mutex<Vec<((&'static str, String), u64)>> = Mutex::new(Vec::new());

/// Whether this process observes the capture's steps.
///
/// Read once, and the read is announced: a boot whose log has capture lines and
/// no `capture_probe on=…` line ran without the probe, which is a different
/// reading from a boot whose probe counted zero failures.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let (state, value) = crate::config::read(crate::config::CAPTURE_PROBE);
        // Off unless spelled affirmatively. Unset, `off` and a typo all mean
        // "do not observe", for the reason the switch's own doc gives: a probe
        // that is on by accident is a probe whose reading nobody asked for.
        let on = matches!(state, crate::config::Switch::On);
        crate::observe::off(format!(
            "capture_probe on={on} switch={state:?} value={}",
            value.unwrap_or_else(|| "<unset>".into())
        ));
        on
    })
}

/// Count one attempt and the source that served it.
pub fn note_source(source: Source) {
    if !enabled() {
        return;
    }
    let counts = &COUNTS;
    let attempts = counts.attempts.fetch_add(1, Ordering::Relaxed) + 1;
    match source {
        Source::HostCache => counts.host_cache.fetch_add(1, Ordering::Relaxed),
        Source::Resident => counts.resident.fetch_add(1, Ordering::Relaxed),
        Source::None => counts.fail.fetch_add(1, Ordering::Relaxed),
    };
    // Power-of-two spacing: readable from the first attempt, bounded by log2.
    if attempts.is_power_of_two() {
        crate::observe::off(summary_line(attempts));
    }
}

/// Count a capture attempt that found the host cache's *cession* rather than a
/// hit or an empty cache: an entry at this geometry that holds no bytes because
/// the resident is the surface's source.
///
/// It is not a source and not a failure — the attempt goes on to the resident
/// either way — which is why it is a counter of its own rather than folded into
/// either. On the frozen-window reading these are the presents whose host cache
/// *cannot* be the stale frame, because it holds nothing at all.
pub fn note_ceded() {
    if !enabled() {
        return;
    }
    COUNTS.host_cache_ceded.fetch_add(1, Ordering::Relaxed);
}

/// Count the step a failed attempt refused at, and give that signature its
/// detail line the first time it appears.
///
/// `signature` is the identity of the failure for deduplication; `detail`
/// builds the line and is only called when the signature is new, so a repeated
/// failure pays neither the registry report nor the formatting.
pub fn note_step(refusal: CaptureRefusal, signature: &str, detail: impl FnOnce() -> String) {
    if !enabled() {
        return;
    }
    COUNTS.steps[step_index(refusal)].fetch_add(1, Ordering::Relaxed);
    let Ok(mut detailed) = DETAILED.lock() else {
        return;
    };
    // The refusal is part of the key: the same surface refused for two
    // different reasons is two findings, and a key that dropped it would report
    // the second as a repeat of the first.
    let key = (refusal.slug(), signature.to_owned());
    if let Some((_, count)) = detailed.iter_mut().find(|(held, _)| *held == key) {
        *count += 1;
        return;
    }
    if detailed.len() >= MAX_DETAILS {
        return;
    }
    detailed.push((key, 1));
    drop(detailed);
    crate::observe::off(format!(
        "capture_probe reason={} {signature} {}",
        refusal.slug(),
        detail()
    ));
}

/// Count a presented frame's refusal, with the registry key it happened under
/// and, for a new signature, the registry's own answer.
///
/// `key` is what `ResourcePools::registry_key_divergence` says about the key
/// that just missed — how the closest held key differs, and the generation it
/// was held under — and is `None` for a refusal whose key *was* present, where
/// there is no divergence to report and counting one would put every
/// `content_not_ready` into `present_key_*`. It is the parameter this function
/// exists for: the class alone cannot separate a missing target from a key
/// fault, and those are the two states the switch's own doc says have different
/// repairs.
///
/// `signature` identifies the failure for deduplication and `detail` builds the
/// line, so a repeated refusal pays neither the registry report nor the
/// formatting. Bounded by [`PRESENT_MAX_DETAILS`], and the summary still carries
/// the counts past that.
pub fn note_present_refusal(
    refusal: PresentRefusal,
    key: Option<(
        crate::backend::vulkan::engine::TargetKeyDivergence,
        Option<u64>,
    )>,
    signature: &str,
    detail: impl FnOnce() -> String,
) {
    if !enabled() {
        return;
    }
    let counts = &PRESENT_COUNTS;
    let total = counts.refusals.fetch_add(1, Ordering::Relaxed) + 1;
    let slot = PresentRefusal::ALL
        .iter()
        .position(|held| *held == refusal)
        .expect("every present refusal has a slot the vocabulary derives");
    counts.classes[slot].fetch_add(1, Ordering::Relaxed);
    let (key_word, held) = match key {
        None => ("not_asked".to_owned(), "n/a".to_owned()),
        Some((divergence, held)) => {
            use crate::backend::vulkan::engine::TargetKeyDivergence as How;
            let counter = match divergence {
                How::Absent => &counts.key_absent,
                How::Generation => &counts.key_generation,
                How::Geometry => &counts.key_geometry,
                How::Other => &counts.key_other,
                How::Namespace => &counts.key_namespace,
            };
            counter.fetch_add(1, Ordering::Relaxed);
            (
                divergence.label().to_owned(),
                held.map(|generation| generation.to_string())
                    .unwrap_or_else(|| "none".to_owned()),
            )
        }
    };
    let Ok(mut detailed) = PRESENT_DETAILED.lock() else {
        return;
    };
    // The class is part of the key: one surface refused for two different
    // reasons is two findings, and a key that dropped the class would report
    // the second as a repeat of the first.
    let signature_key = (refusal.slug(), signature.to_owned());
    if let Some((_, count)) = detailed.iter_mut().find(|(held, _)| *held == signature_key) {
        *count += 1;
        return;
    }
    if detailed.len() >= PRESENT_MAX_DETAILS {
        return;
    }
    detailed.push((signature_key, 1));
    drop(detailed);
    crate::observe::off(format!(
        "capture_probe present_refused={} key={key_word} held_generation={held} {signature} {}",
        refusal.slug(),
        detail()
    ));
    // Power-of-two spacing, the same rule the capture side uses: the ratio is
    // readable from the first refusal onward and a hot present cannot flood.
    if total.is_power_of_two() {
        crate::observe::off(summary_line(COUNTS.attempts.load(Ordering::Relaxed)));
    }
}

/// The summary, with every class printed — a zero here is a reading rather than
/// an absence, which is the whole reason the probe has its own line.
fn summary_line(attempts: u64) -> String {
    let counts = &COUNTS;
    let mut out = format!(
        "capture_probe attempts={attempts} host_cache_ceded={} distinct_signatures={}",
        counts.host_cache_ceded.load(Ordering::Relaxed),
        DETAILED.lock().map(|held| held.len()).unwrap_or(0),
    );
    // Named through the source's own slug rather than written out again, so the
    // summary and the counter cannot disagree about what a source is called.
    for (source, count) in [
        (Source::HostCache, &counts.host_cache),
        (Source::Resident, &counts.resident),
        (Source::None, &counts.fail),
    ] {
        out.push_str(&format!(
            " {}={}",
            source.slug(),
            count.load(Ordering::Relaxed)
        ));
    }
    for refusal in CaptureRefusal::ALL {
        out.push_str(&format!(
            " refused_{}={}",
            refusal.slug(),
            counts.steps[step_index(refusal)].load(Ordering::Relaxed)
        ));
    }
    // The publish half, on the same line and under its own prefix. `attempts`
    // above is the capture's, so the two denominators cannot be confused: the
    // presents that reached a publish are `present_refused_total` plus whatever
    // was served, and this line is read beside `host_window_cadence`, which
    // carries the served half.
    let presents = &PRESENT_COUNTS;
    out.push_str(&format!(
        " present_refused_total={} present_signatures={}",
        presents.refusals.load(Ordering::Relaxed),
        PRESENT_DETAILED.lock().map(|held| held.len()).unwrap_or(0),
    ));
    for refusal in PresentRefusal::ALL {
        let slot = PresentRefusal::ALL
            .iter()
            .position(|held| *held == refusal)
            .expect("every present refusal has a slot the vocabulary derives");
        out.push_str(&format!(
            " present_refused_{}={}",
            refusal.slug(),
            presents.classes[slot].load(Ordering::Relaxed)
        ));
    }
    for (label, counter) in [
        ("absent", &presents.key_absent),
        ("generation", &presents.key_generation),
        ("geometry", &presents.key_geometry),
        ("other", &presents.key_other),
        ("namespace", &presents.key_namespace),
    ] {
        out.push_str(&format!(
            " present_key_{label}={}",
            counter.load(Ordering::Relaxed)
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vocabulary is the shared one, and this census must cover all of it:
    /// a refusal that exists and has no slot here would be counted nowhere
    /// while still appearing on the failure channel.
    #[test]
    fn every_refusal_has_its_own_name_and_slot() {
        let mut seen = Vec::new();
        for (index, refusal) in CaptureRefusal::ALL.into_iter().enumerate() {
            assert_eq!(
                step_index(refusal),
                index,
                "{refusal:?} must index its own slot"
            );
            let slug = refusal.slug();
            assert!(!slug.is_empty(), "{refusal:?} needs a name");
            assert!(!seen.contains(&slug), "{slug} names two refusals");
            seen.push(slug);
        }
        assert_eq!(seen.len(), CaptureRefusal::ALL.len());
    }

    /// The three sources are what the caller can observe: which vein served the
    /// frame, or that neither did.
    #[test]
    fn the_three_sources_are_named_apart() {
        assert_eq!(Source::HostCache.slug(), "host_cache");
        assert_eq!(Source::Resident.slug(), "resident");
        assert_eq!(Source::None.slug(), "fail");
        let line = summary_line(1);
        for source in [Source::HostCache, Source::Resident, Source::None] {
            assert!(
                line.contains(&format!(" {}=", source.slug())),
                "{line} must name {} as its own field",
                source.slug()
            );
        }
    }

    /// Off is off: with the switch unset nothing is counted, so a boot that did
    /// not ask for the probe cannot report a reading from it. The process value
    /// is latched, so this asserts the latch rather than an env read.
    #[test]
    fn the_probe_is_off_unless_the_switch_asks_for_it() {
        let (state, _) = crate::config::read(crate::config::CAPTURE_PROBE);
        assert!(
            !matches!(state, crate::config::Switch::On),
            "this test runs in a process whose env must not set {}",
            crate::config::CAPTURE_PROBE
        );
        assert!(!enabled(), "the probe must be off without the switch");
        let before = summary_line(1);
        note_source(Source::None);
        note_step(CaptureRefusal::NoTarget, "mid=1", || "unused".to_owned());
        assert_eq!(
            summary_line(1),
            before,
            "a probe that is off must count nothing"
        );
    }
}
