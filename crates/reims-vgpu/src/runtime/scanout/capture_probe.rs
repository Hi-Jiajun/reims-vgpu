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
