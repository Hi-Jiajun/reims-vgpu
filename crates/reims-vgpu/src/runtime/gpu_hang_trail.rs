//! What this device had just asked the GPU to do, kept so that a hang can be
//! named.
//!
//! # The gap this closes
//!
//! When the host GPU wedges, the kernel logs `GPU HANG: ecode …` and resets the
//! context, and this device sees a fence that never signals. By then the batch
//! that hung is gone: `wait_error` knows only that a wait timed out, and every
//! refusal reason downstream of it — `frame_bgra_short`, `no_resident_content`,
//! `read_target_unknown_identity` — is a consequence of the device loss rather
//! than a description of what caused it. Two full sessions have looked for the
//! cause by bisecting the device's narrowing switches, which is a way of asking
//! *whether a rail* is responsible and not *which submission* was.
//!
//! Nor does the kernel's own line say. Its process name identifies the thread
//! that created the i915 context, not the one that submitted the batch — see
//! `kb/the-comm-in-a-gpu-hang-line-names-who-created-the-context.md`.
//!
//! So this keeps the last few pieces of work this device recorded, in memory,
//! and prints them when a drain tranche is caught holding the engine past
//! [`crate::runtime::drain::SYNC_EXEC_STALL_US`] — which on the boots measured
//! so far is the moment the engine wedges, and which fires on the drain thread
//! while it is blocked, so the tail of the trail is the work it is blocked on.
//!
//! # Why a fixed ring and not a counter
//!
//! A counter answers "how much", and the question here is "which". A pipeline
//! that hangs does so because of what it *is* — its fragment module, its
//! geometry, its instance count — and none of that survives as a number
//! averaged over a second. [`CAPACITY`] entries is the bound, chosen so the
//! whole ring is one log line: a hang holds the engine for seconds and the drain
//! blocks inside it, so the work that matters is the last handful and not the
//! last thousand.
//!
//! It is deliberately *not* gated behind [`crate::env::DRAW_LOG`]. That switch
//! turns on a per-draw log flood, which is itself a drain cost heavy enough to
//! change what it measures; this writes seven integers into a fixed array and
//! prints nothing until something has already gone wrong.
//!
//! # Only the Vulkan draw path writes it, and that is not an oversight
//!
//! Two of the fields — the translated module word counts — do not exist on the
//! Metal-direct arm, where the guest's own AIR is handed to the Metal compiler
//! and never becomes SPIR-V. A Metal producer would have to record different
//! fields for a different failure, so it belongs in its own trail if a Metal
//! host is ever seen wedging, not in this one with two columns zeroed. The type
//! and its reader are ungated so that arm compiles; on it the trail stays empty
//! and [`trail`] answers `None`, which emits no line.

use std::sync::Mutex;

/// Entries kept. One log line's worth.
const CAPACITY: usize = 12;

/// One piece of work this device recorded for the GPU.
///
/// The fields are what tells one hanging candidate from another: which guest
/// pipeline object it was, how large each of its two translated modules is —
/// the discriminator for a compositing uber shader against an ordinary blit —
/// and how much geometry it asked for.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DrawNote {
    pub pipeline_ref: u32,
    pub vert_words: u32,
    pub frag_words: u32,
    pub width: u32,
    pub height: u32,
    pub vertex_count: u32,
    pub instance_count: u32,
}

impl std::fmt::Display for DrawNote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "pipe={} vw={} fw={} {}x{} vtx={} inst={}",
            self.pipeline_ref,
            self.vert_words,
            self.frag_words,
            self.width,
            self.height,
            self.vertex_count,
            self.instance_count
        )
    }
}

/// The ring. A `Mutex` rather than a lock-free structure because every writer is
/// the drain worker and the one reader is the same thread inside a stall it has
/// already lost seconds to.
static TRAIL: Mutex<Trail> = Mutex::new(Trail {
    notes: [None; CAPACITY],
    next: 0,
    total: 0,
});

struct Trail {
    notes: [Option<DrawNote>; CAPACITY],
    next: usize,
    /// Every note ever recorded, so a trail can say how much it is *not*
    /// showing. A trail of twelve out of twelve is the whole boot; twelve out of
    /// four hundred thousand is a tail.
    total: u64,
}

/// Record one draw this device is about to hand the engine.
pub fn note_draw(note: DrawNote) {
    let mut trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    let slot = trail.next;
    trail.notes[slot] = Some(note);
    trail.next = (slot + 1) % CAPACITY;
    trail.total = trail.total.wrapping_add(1);
}

/// The trail, oldest first, as one line's worth of text.
///
/// `None` when nothing has been recorded, so a caller emits no line rather than
/// an empty one — a stall with no draws behind it is a real and different
/// reading from a stall whose draws are unremarkable, and an empty list would
/// spell them the same way.
pub fn trail() -> Option<String> {
    let trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    if trail.total == 0 {
        return None;
    }
    let kept = trail.notes.iter().filter(|n| n.is_some()).count();
    // Oldest first: start at the write cursor, which is the oldest live slot
    // once the ring has wrapped and an empty one before that.
    let body = (0..CAPACITY)
        .filter_map(|i| trail.notes[(trail.next + i) % CAPACITY].as_ref())
        .map(|n| format!("[{n}]"))
        .collect::<Vec<_>>()
        .join(" ");
    Some(format!("kept={kept}/{} {body}", trail.total))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(pipeline_ref: u32) -> DrawNote {
        DrawNote {
            pipeline_ref,
            ..DrawNote::default()
        }
    }

    /// Nothing recorded is not the same reading as nothing interesting
    /// recorded, so the caller gets to say so.
    #[test]
    fn an_empty_trail_reports_nothing_rather_than_an_empty_list() {
        // Shares the process-wide ring with the tests below, so this one asserts
        // only what is true whatever they have written: a trail is `Some` once
        // anything has been noted, and this test cannot un-note it.
        if trail().is_none() {
            note_draw(note(1));
            assert!(trail().is_some(), "a note makes the trail readable");
        }
    }

    /// The ring keeps the *newest* entries and reports them oldest first, which
    /// is the order a reader reconstructs a submission sequence in.
    #[test]
    fn the_ring_keeps_the_newest_entries_in_arrival_order() {
        for i in 0..(CAPACITY as u32 * 2) {
            note_draw(note(1000 + i));
        }
        let line = trail().expect("notes were recorded");
        let first = 1000 + CAPACITY as u32;
        let last = 1000 + CAPACITY as u32 * 2 - 1;
        let pipes: Vec<&str> = line.match_indices("pipe=").map(|(i, _)| &line[i..]).collect();
        assert_eq!(pipes.len(), CAPACITY, "the ring is full: {line}");
        assert!(
            line.contains(&format!("pipe={first} ")),
            "the oldest kept entry is the first of the last {CAPACITY}: {line}"
        );
        assert!(
            line.contains(&format!("pipe={last} ")),
            "the newest entry is kept: {line}"
        );
        assert!(
            !line.contains(&format!("pipe={} ", first - 1)),
            "the entry before the window was evicted: {line}"
        );
        let first_at = line.find(&format!("pipe={first} ")).unwrap();
        let last_at = line.find(&format!("pipe={last} ")).unwrap();
        assert!(first_at < last_at, "oldest first: {line}");
    }

    /// The total is what says whether a full ring is the whole boot or its tail.
    #[test]
    fn the_trail_says_how_much_it_is_not_showing() {
        for i in 0..(CAPACITY as u32 + 5) {
            note_draw(note(2000 + i));
        }
        let line = trail().expect("notes were recorded");
        let kept = line
            .split_whitespace()
            .next()
            .and_then(|f| f.strip_prefix("kept="))
            .and_then(|f| f.split_once('/'))
            .map(|(k, t)| (k.to_string(), t.to_string()))
            .expect("the line leads with kept=N/M");
        assert_eq!(kept.0, CAPACITY.to_string());
        assert!(
            kept.1.parse::<u64>().expect("a total") > CAPACITY as u64,
            "the total counts every note, not the kept ones: {line}"
        );
    }
}
