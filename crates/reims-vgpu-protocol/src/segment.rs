//! What a segment-type byte means.
//!
//! # Why the meaning lives here and the bytes live in wire
//!
//! `reims_vgpu_wire::ops::segment` owns the header layout and the four type
//! values its fixtures measured, plus the protection-options envelope. It
//! cannot say that type `1` "is a compute encoder whose records are read on the
//! compute rail", because that is a meaning and wire does not assign meanings.
//! This module does, and it takes every value it can from wire rather than
//! restating a number, so a fixture that moved one would break the parse rather
//! than leaving two disagreeing copies.
//!
//! # The one value wire does not name
//!
//! Wire has driven the render, compute, blit and info encoders and measured
//! `0`, `1`, `2` and `4`. It deliberately does not name `3`: no fixture wrote
//! it. Two independent facts establish it anyway. The deserializer constructs
//! record decoders for the contiguous set `0..=3` and rejects new
//! non-continuation types at `4` and above, so `3` is a constructed decoder
//! rather than a hole; and the remaining encoder class in that set is the event
//! encoder, which [`crate::closure`] now carries as [`Rail::Event`] with its own
//! records. So `3` is named here, at the layer that assigns meaning, with the
//! derivation attached — and not in wire, which would have to pretend a fixture
//! wrote it.
//!
//! # Unknown is a refusal, not a variant
//!
//! [`segment_role`] returns `None` for every other byte. A segment whose family
//! is unknown has an unknown record framing, so walking it reads guest data as
//! commands; and a catch-all variant would hand it ordering and a completion
//! obligation the device cannot honour. The caller refuses.

use crate::closure::Rail;
use reims_vgpu_wire::ops::segment::{
    SEGMENT_TYPE_BLIT, SEGMENT_TYPE_COMPUTE, SEGMENT_TYPE_INFO, SEGMENT_TYPE_PROTECTION_OPTIONS,
    SEGMENT_TYPE_RENDER,
};

/// The event encoder's segment type.
///
/// Derived rather than measured; see the module documentation.
pub const SEGMENT_TYPE_EVENT: u8 = 3;

/// A record-bearing segment: one encoder's worth of commands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SegmentKind {
    Render,
    Compute,
    Blit,
    Event,
    Info,
}

impl SegmentKind {
    pub const ALL: &'static [SegmentKind] = &[
        SegmentKind::Render,
        SegmentKind::Compute,
        SegmentKind::Blit,
        SegmentKind::Event,
        SegmentKind::Info,
    ];

    /// The byte this family writes into the segment header.
    #[must_use]
    pub const fn wire_type(self) -> u8 {
        match self {
            Self::Render => SEGMENT_TYPE_RENDER,
            Self::Compute => SEGMENT_TYPE_COMPUTE,
            Self::Blit => SEGMENT_TYPE_BLIT,
            Self::Event => SEGMENT_TYPE_EVENT,
            Self::Info => SEGMENT_TYPE_INFO,
        }
    }

    /// The rail whose records may appear inside this segment.
    ///
    /// One-to-one, and that is the point: a rail is how an opcode is read, a
    /// segment kind is where the record was found, and a model that trusted
    /// either alone would read one encoder's commands as another's.
    #[must_use]
    pub const fn rail(self) -> Rail {
        match self {
            Self::Render => Rail::Render,
            Self::Compute => Rail::Compute,
            Self::Blit => Rail::Blit,
            Self::Event => Rail::Event,
            Self::Info => Rail::Info,
        }
    }

    /// The segment kind whose records are read on `rail`, if any.
    ///
    /// [`Rail::Root`] has none: its records arrive as object-list payloads and
    /// never inside a command stream.
    #[must_use]
    pub const fn of_rail(rail: Rail) -> Option<SegmentKind> {
        Some(match rail {
            Rail::Render => Self::Render,
            Rail::Compute => Self::Compute,
            Rail::Blit => Self::Blit,
            Rail::Event => Self::Event,
            Rail::Info => Self::Info,
            Rail::Root => return None,
        })
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Render => "render",
            Self::Compute => "compute",
            Self::Blit => "blit",
            Self::Event => "event",
            Self::Info => "info",
        }
    }
}

/// What a segment header introduces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SegmentRole {
    /// An encoder's records follow.
    Encoder(SegmentKind),
    /// A protection-options envelope, which arms the *next* encoder segment and
    /// carries no records of its own.
    ProtectionEnvelope,
}

/// Parse a segment-type byte, or refuse it.
#[must_use]
pub const fn segment_role(wire_type: u8) -> Option<SegmentRole> {
    Some(match wire_type {
        SEGMENT_TYPE_RENDER => SegmentRole::Encoder(SegmentKind::Render),
        SEGMENT_TYPE_COMPUTE => SegmentRole::Encoder(SegmentKind::Compute),
        SEGMENT_TYPE_BLIT => SegmentRole::Encoder(SegmentKind::Blit),
        SEGMENT_TYPE_EVENT => SegmentRole::Encoder(SegmentKind::Event),
        SEGMENT_TYPE_INFO => SegmentRole::Encoder(SegmentKind::Info),
        SEGMENT_TYPE_PROTECTION_OPTIONS => SegmentRole::ProtectionEnvelope,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kind round-trips, and nothing outside the six established values
    /// parses at all.
    #[test]
    fn the_byte_map_is_a_bijection_over_the_established_values() {
        for &kind in SegmentKind::ALL {
            assert_eq!(
                segment_role(kind.wire_type()),
                Some(SegmentRole::Encoder(kind))
            );
        }
        assert_eq!(
            segment_role(SEGMENT_TYPE_PROTECTION_OPTIONS),
            Some(SegmentRole::ProtectionEnvelope)
        );
        for byte in 0u8..=255 {
            let established = SegmentKind::ALL.iter().any(|k| k.wire_type() == byte)
                || byte == SEGMENT_TYPE_PROTECTION_OPTIONS;
            assert_eq!(segment_role(byte).is_some(), established, "{byte:#x}");
        }
    }

    /// The values are wire's, not this module's — with `3` the stated
    /// exception. A test rather than a comment, because the failure mode is two
    /// copies drifting silently.
    #[test]
    fn the_measured_values_come_from_wire() {
        assert_eq!(SegmentKind::Render.wire_type(), SEGMENT_TYPE_RENDER);
        assert_eq!(SegmentKind::Compute.wire_type(), SEGMENT_TYPE_COMPUTE);
        assert_eq!(SegmentKind::Blit.wire_type(), SEGMENT_TYPE_BLIT);
        assert_eq!(SegmentKind::Info.wire_type(), SEGMENT_TYPE_INFO);
        assert_eq!(SegmentKind::Event.wire_type(), 3);
    }

    /// `Info` is `4`, not the next integer after blit. A device that had
    /// guessed the sequence would read every info segment as an event one.
    #[test]
    fn info_is_four_and_the_gap_belongs_to_events() {
        assert_eq!(SEGMENT_TYPE_INFO, 4);
        assert_eq!(
            segment_role(3),
            Some(SegmentRole::Encoder(SegmentKind::Event))
        );
    }

    /// Rails and segments correspond one-to-one, except the root rail, which
    /// has no segment because its records never enter a command stream.
    #[test]
    fn rails_and_segments_correspond_except_at_the_root() {
        for &kind in SegmentKind::ALL {
            assert_eq!(SegmentKind::of_rail(kind.rail()), Some(kind));
            assert_ne!(kind.rail(), Rail::Root);
        }
        assert_eq!(SegmentKind::of_rail(Rail::Root), None);
    }
}
