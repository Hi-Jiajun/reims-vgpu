//! Always-on proxy for the sRGB-downgrade bug class.
//!
//! # The class
//!
//! The guest names an sRGB render target or texture; the host binds the linear
//! sibling of that format. The hardware then never applies the sRGB transfer
//! function, so blending and sampling happen in the wrong colour space. The
//! defect is not that the fold exists — several rails genuinely carry raw bytes
//! and cannot encode — it is that the fold used to be **silent**: a lost format
//! qualifier looked exactly like a supported format, with nothing in the fail
//! log.
//!
//! # A zero here does not mean no rail dropped a qualifier
//!
//! Read this before excluding the sRGB family on the strength of an empty
//! census, because a session already did and was wrong.
//!
//! This census reports the sites listed in [`site`] and nothing else. It once
//! carried three more — the linear, type-11 and type-5 **zero-copy** sampled
//! rails — which were removed when those rails were changed to bind the
//! `_SRGB` view and honour the qualifier. They had no emitter for some time
//! before that, so the census was answering for a population it no longer
//! watched.
//!
//! The rail that used to be missing here was the **CPU sampled upload rung**,
//! `runtime::draw::vulkan`'s `SampledSourceRequest::Bytes` arm. It could not be
//! given a site constant, because a site needs the guest's declared format at
//! the moment of the fold and that variant carried a bare `TexelLayout` — into
//! which the qualifier had already been narrowed away by the producer. So the
//! rung downgraded every sRGB CPU upload silently, while the zero-copy rails
//! beside it bound the `_SRGB` view and decoded, and which of the two a bind
//! took was a cost decision.
//!
//! That variant now carries [`crate::contract::pixel_format::SampledByteFormat`],
//! which pairs the layout with the source format, so the rung *honours* the
//! qualifier and there is nothing left to report for the eight-bit colour
//! orders. What remains is [`site::SAMPLED_BYTE_UPLOAD`]: a loader that
//! converted an sRGB texture into a layout with no sRGB spelling has moved the
//! values out of the encoding's domain, and that is a genuine loss this can
//! finally name.
//!
//! # Reading it
//!
//! `/tmp/reims-vgpu-fail.log`, always-on, one line per (site, format) pair:
//!
//! * `srgb_downgraded reason=srgb_downgraded site=<site> mtl=<fmt> …`
//!
//! **No lines on a healthy boot means the guest never asked for sRGB.** A line
//! is not itself a failure; it says which rail is trading colour-space
//! correctness away, which is the only thing needed to decide where adopting
//! `VK_FORMAT_*_SRGB` would pay. The pair is the unit because a rail hit twice
//! with the same format has nothing more to say the second time.
//!
//! Measure-only: nothing here gates decode, execute or present.

use std::collections::BTreeSet;
use std::sync::Mutex;

use crate::observe;

/// The slug every downgrade line carries. Kept equal to
/// `TranslateReason::SrgbDowngraded`'s slug by a unit test in the Vulkan
/// backend, so the typed reason and the always-on line cannot drift apart.
pub const SRGB_DOWNGRADED_SLUG: &str = "srgb_downgraded";

/// The rails that can drop an sRGB qualifier, each named for the code path a
/// reader would open. One constant per site so a log line points somewhere
/// specific rather than at "the sampled path".
pub mod site {
    /// `build_secondary_targets` — MRT colour attachment beyond slot 0.
    pub const SECONDARY_COLOR_TARGET: &str = "secondary_color_target";
    /// `translate::pixel::vk_sampled_bytes` — a CPU loader handed back bytes
    /// that are still sRGB-encoded in a layout with no sRGB spelling, so the
    /// linear one is bound and the hardware will not decode.
    ///
    /// The CPU upload rails do not otherwise appear here any more: they carry
    /// the source format alongside the layout and the fold honours it. This
    /// fires only where honouring it is not expressible.
    pub const SAMPLED_BYTE_UPLOAD: &str = "sampled_byte_upload";

    /// Every site, for the completeness test. A new site constant that is not
    /// listed here is one the census cannot report on.
    pub const ALL: &[&str] = &[SECONDARY_COLOR_TARGET, SAMPLED_BYTE_UPLOAD];
}

/// `(site, MTLPixelFormat)` pairs already reported, so a per-draw rail costs one
/// line per distinct pair per boot. Bounded by `site::ALL` times the small set
/// of sRGB formats.
static SEEN: Mutex<BTreeSet<(&'static str, u16)>> = Mutex::new(BTreeSet::new());

/// Record that `site` bound the linear sibling of the sRGB format `mtl`.
///
/// Call this at the moment the qualifier is dropped, not at the moment the
/// format is decoded — the point is to name what was actually traded away.
pub fn note_downgrade(site: &'static str, mtl: u16) {
    let first_sight = SEEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert((site, mtl));
    if first_sight {
        observe::fail(format!(
            "srgb_downgraded reason={SRGB_DOWNGRADED_SLUG} site={site} mtl={mtl:#x} \
             (bound the linear sibling; hardware will not apply the sRGB transfer \
             function on this rail)"
        ));
    }
}

/// Drop the first-sight set. Test-only: it is process-global, so a test that
/// asserts a line was emitted must start from a known point.
#[cfg(test)]
pub fn reset_for_tests() {
    SEEN.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::pixel_format::{MTL_FORMAT_BGRA8_UNORM_SRGB, MTL_FORMAT_RGBA8_UNORM_SRGB};

    /// A per-draw rail must cost one line per distinct (site, format) pair, not
    /// one per bind — the dedup is what makes it safe to leave on forever, and
    /// a second pair on the same site is still a new event.
    #[test]
    fn each_site_and_format_pair_reports_once() {
        reset_for_tests();
        assert!(SEEN
            .lock()
            .unwrap()
            .insert((site::SAMPLED_BYTE_UPLOAD, MTL_FORMAT_BGRA8_UNORM_SRGB)));
        for _ in 0..64 {
            note_downgrade(site::SAMPLED_BYTE_UPLOAD, MTL_FORMAT_BGRA8_UNORM_SRGB);
        }
        assert_eq!(SEEN.lock().unwrap().len(), 1, "64 binds, one pair");
        note_downgrade(site::SAMPLED_BYTE_UPLOAD, MTL_FORMAT_RGBA8_UNORM_SRGB);
        note_downgrade(site::SECONDARY_COLOR_TARGET, MTL_FORMAT_RGBA8_UNORM_SRGB);
        assert_eq!(
            SEEN.lock().unwrap().len(),
            3,
            "a new format and a new site are each a new event"
        );
        reset_for_tests();
    }

    /// Site names are distinct and log-safe — a duplicate would merge two
    /// rails' counts and a space would break the field split.
    #[test]
    fn site_names_are_distinct_and_log_safe() {
        let mut names = site::ALL.to_vec();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate site name");
        for name in site::ALL {
            assert!(name
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'));
        }
    }
}
