//! The query transaction: a question the guest blocks on, and the one ordering
//! rule that makes an answer an answer.
//!
//! # Why a query is not a control command
//!
//! The four query packets carry a reply destination and the guest waits on the
//! completion word before reading it. The ledger's reading of one of them says
//! it exactly: *a query whose reply is written before the stamp retires,
//! because the guest blocks on it — a compute pipeline creation stalls without
//! the answer.* So an unanswered query is not lost work the guest can retry. It
//! is a **wrong answer**: the guest reads whatever the destination already held
//! and proceeds on it.
//!
//! That gives the class two obligations no other payload has. The reply must be
//! visible before the completion word, and the completion word must be
//! published *even when there is no answer* — because a query that neither
//! answers nor completes is a hang, which is strictly worse than an answer the
//! guest can at least be told about on the failure channel.
//!
//! # The order is a type, not a comment
//!
//! [`PendingQuery`] holds the stamp and has no way to hand it over.
//! [`CompletedQuery`] has the stamp and can only be reached through
//! [`PendingQuery::answer`] or [`PendingQuery::unanswerable`] — so every path
//! to publication has already either produced the reply write or named why
//! there is none. "Write the reply before the stamp" is therefore not a rule a
//! reviewer checks; there is no second path to write.
//!
//! `PendingQuery` is `#[must_use]` for the remaining hole: dropping one is the
//! hang, and it is the one thing the types cannot forbid.
//!
//! # What this does not decide
//!
//! What the answer *is*. Device limits, threadgroup sizes and heap texture
//! geometry are host capabilities, and this crate cannot see a host. It owns
//! where the answer goes, how much of it fits, and when it becomes visible
//! relative to the completion word — the layout of the common reply shape being
//! [`reims_vgpu_protocol::info_reply`]'s.

use crate::access::{BackingId, ByteRange};
use crate::identity::CompletionStamp;
use crate::transaction::{classify, PayloadClass};
use reims_vgpu_protocol::packets::Channel;

/// Which question a query packet asks.
///
/// Exhaustive over the packet classes [`classify`] calls
/// [`PayloadClass::Query`], which is what the totality test checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QueryKind {
    /// The Monterey-era device-info query, still answered for a guest old
    /// enough to ask.
    DeviceInfoLegacy,
    /// The current device-info query.
    DeviceInfo,
    /// A compute pipeline's threadgroup limits.
    ComputeInfo,
    /// A heap texture's size and alignment.
    HeapTextureSizeAndAlign,
}

impl QueryKind {
    /// The question a packet asks, or `None` if it is not a query packet.
    #[must_use]
    pub fn of(channel: Channel, opcode: u16) -> Option<Self> {
        if classify(channel, opcode) != Some(PayloadClass::Query) {
            return None;
        }
        Some(match (channel, opcode) {
            (Channel::Root, 0x2d) => Self::DeviceInfoLegacy,
            (Channel::Root, 0x3a) => Self::DeviceInfo,
            (Channel::Child, 0x3b) => Self::ComputeInfo,
            (Channel::Child, 0x40) => Self::HeapTextureSizeAndAlign,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DeviceInfoLegacy => "device_info_legacy",
            Self::DeviceInfo => "device_info",
            Self::ComputeInfo => "compute_info",
            Self::HeapTextureSizeAndAlign => "heap_texture_size_and_align",
        }
    }
}

/// Where the answer goes, already resolved.
///
/// A backing and a window of it, not a guest address: resolving the address the
/// request names is the caller's, and a destination that arrived here as an
/// address would be one this crate could not bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplyDestination {
    pub backing: BackingId,
    pub bytes: ByteRange,
}

/// The question a query packet asks, and where its answer goes.
///
/// The two travel together because neither is a query on its own: a kind with
/// no destination is an answer with nowhere to be written, and a destination
/// with no kind is a window with nothing to put in it. The guest blocks on the
/// pair.
///
/// The completion stamp is deliberately *not* here — it is the envelope's, and
/// [`PendingQuery`] takes it from there when the packet is admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryRequest {
    pub kind: QueryKind,
    pub destination: ReplyDestination,
}

/// The exact window the answer occupies.
///
/// The caller records it as a write in the guest's replica: the answer is
/// content the guest is about to read, and content the ledger does not know
/// about is content a later transfer can overwrite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "an answer the content authority does not know about can be overwritten by a transfer"]
pub struct ReplyWrite {
    pub backing: BackingId,
    pub bytes: ByteRange,
}

/// Why a query was completed without an answer.
///
/// Every variant is a stall the guest will experience as a wrong value, so each
/// one is named rather than folded into a generic failure: which of them
/// happened decides whether the defect is in the request, in the destination,
/// or in this device's ability to answer at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stall {
    /// The answer does not fit in the destination the request named. Not
    /// clamped: a partial reply is a reply, and the guest cannot tell it from a
    /// complete one.
    ReplyTooLarge { needed: u64, available: u64 },
    /// This device has no answer for the question. The stamp is still
    /// published, so the guest proceeds on its own zeroed destination rather
    /// than waiting forever.
    NoAnswer,
}

impl Stall {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::ReplyTooLarge { .. } => "query_reply_too_large",
            Self::NoAnswer => "query_no_answer",
        }
    }
}

/// A query that has been admitted and not yet answered.
///
/// Holds the completion stamp and cannot give it up. The only ways out are
/// [`Self::answer`] and [`Self::unanswerable`], which is what makes
/// "the reply is written before the stamp" structural.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "a query that is neither answered nor completed is a guest that blocks forever"]
pub struct PendingQuery {
    kind: QueryKind,
    destination: ReplyDestination,
    stamp: Option<CompletionStamp>,
}

/// How a query ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Answer {
    /// The reply was written to exactly this window.
    Written(ReplyWrite),
    /// There is no reply, and this is why.
    None(Stall),
}

/// A query whose answer — or lack of one — is settled, and whose completion
/// word may therefore be published.
#[derive(Debug, PartialEq, Eq)]
pub struct CompletedQuery {
    kind: QueryKind,
    answer: Answer,
    stamp: Option<CompletionStamp>,
}

impl PendingQuery {
    pub const fn new(request: QueryRequest, stamp: Option<CompletionStamp>) -> Self {
        Self {
            kind: request.kind,
            destination: request.destination,
            stamp,
        }
    }

    #[must_use]
    pub const fn kind(&self) -> QueryKind {
        self.kind
    }

    #[must_use]
    pub const fn destination(&self) -> ReplyDestination {
        self.destination
    }

    /// How many bytes the destination holds, for an evaluator sizing its reply.
    #[must_use]
    pub const fn capacity(&self) -> u64 {
        self.destination.bytes.length
    }

    /// Record that an answer of `len` bytes was written at the destination.
    ///
    /// Consumes the pending state: after this the completion word may be
    /// published and the reply cannot be written again.
    ///
    /// # Errors
    ///
    /// If the answer does not fit. The query is returned with the pending
    /// state intact so the caller can complete it as unanswerable — a stall
    /// that was refused still has to publish its stamp.
    pub fn answer(self, len: u64) -> Result<CompletedQuery, (Self, Stall)> {
        if len > self.destination.bytes.length {
            let stall = Stall::ReplyTooLarge {
                needed: len,
                available: self.destination.bytes.length,
            };
            return Err((self, stall));
        }
        Ok(CompletedQuery {
            kind: self.kind,
            answer: Answer::Written(ReplyWrite {
                backing: self.destination.backing,
                bytes: ByteRange {
                    offset: self.destination.bytes.offset,
                    length: len,
                },
            }),
            stamp: self.stamp,
        })
    }

    /// Complete the query with no answer, for a named reason.
    ///
    /// Not an error path to be avoided: the stamp is published either way,
    /// because a guest blocked on a completion word it never receives is worse
    /// than a guest that reads a value it can be told was never written.
    pub fn unanswerable(self, reason: Stall) -> CompletedQuery {
        CompletedQuery {
            kind: self.kind,
            answer: Answer::None(reason),
            stamp: self.stamp,
        }
    }
}

impl CompletedQuery {
    #[must_use]
    pub const fn kind(&self) -> QueryKind {
        self.kind
    }

    #[must_use]
    pub const fn answer(&self) -> Answer {
        self.answer
    }

    /// The reply's window, if there was a reply.
    #[must_use]
    pub const fn write(&self) -> Option<ReplyWrite> {
        match self.answer {
            Answer::Written(w) => Some(w),
            Answer::None(_) => None,
        }
    }

    /// Why there is no reply, if there is none.
    #[must_use]
    pub const fn stall(&self) -> Option<Stall> {
        match self.answer {
            Answer::None(s) => Some(s),
            Answer::Written(_) => None,
        }
    }

    /// The completion word this query owes, now that its answer is settled.
    #[must_use]
    pub const fn publication(&self) -> Option<CompletionStamp> {
        self.stamp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{StampSlot, StampValue};
    use reims_vgpu_protocol::info_reply::{self, ReplyBounds};
    use reims_vgpu_protocol::packets::LEDGER;

    fn stamp() -> CompletionStamp {
        CompletionStamp {
            slot: StampSlot(4),
            value: StampValue(9),
        }
    }

    fn destination(length: u64) -> ReplyDestination {
        ReplyDestination {
            backing: BackingId(3),
            bytes: ByteRange {
                offset: 128,
                length,
            },
        }
    }

    fn pending(length: u64) -> PendingQuery {
        PendingQuery::new(
            QueryRequest {
                kind: QueryKind::ComputeInfo,
                destination: destination(length),
            },
            Some(stamp()),
        )
    }

    /// The claim the module docs make and cannot check by being read.
    #[test]
    fn every_query_packet_has_exactly_one_kind() {
        let mut seen: Vec<QueryKind> = Vec::new();
        for p in LEDGER {
            let kind = QueryKind::of(p.channel, p.opcode);
            let is_query = classify(p.channel, p.opcode) == Some(PayloadClass::Query);
            assert_eq!(
                kind.is_some(),
                is_query,
                "{} {:#04x} is classified {:?} and resolves to {:?}",
                p.channel.name(),
                p.opcode,
                classify(p.channel, p.opcode),
                kind
            );
            if let Some(k) = kind {
                seen.push(k);
            }
        }
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 4, "four questions, and no fifth kind");
    }

    #[test]
    fn an_answer_names_exactly_the_window_it_occupies() {
        let done = pending(4096).answer(24).expect("fits");
        assert_eq!(
            done.write(),
            Some(ReplyWrite {
                backing: BackingId(3),
                bytes: ByteRange {
                    offset: 128,
                    length: 24
                },
            }),
            "at the destination's offset, and only as long as the answer"
        );
        assert_eq!(done.stall(), None);
        assert_eq!(done.publication(), Some(stamp()));
    }

    /// A partial reply is indistinguishable from a complete one to the guest,
    /// so it is refused rather than clamped.
    #[test]
    fn an_answer_that_does_not_fit_is_refused_and_not_clamped() {
        let (still_pending, stall) = pending(16).answer(24).expect_err("does not fit");
        assert_eq!(
            stall,
            Stall::ReplyTooLarge {
                needed: 24,
                available: 16
            }
        );
        // And the refusal still has to complete, or the guest blocks forever.
        let done = still_pending.unanswerable(stall);
        assert_eq!(done.write(), None);
        assert_eq!(done.stall(), Some(stall));
        assert_eq!(
            done.publication(),
            Some(stamp()),
            "an unanswered query publishes its stamp; a hang is worse than a \
             value the guest can be told about"
        );
    }

    #[test]
    fn a_query_with_no_completion_word_publishes_nothing() {
        let done = PendingQuery::new(
            QueryRequest {
                kind: QueryKind::DeviceInfo,
                destination: destination(64),
            },
            None,
        )
        .answer(8)
        .expect("fits");
        assert_eq!(done.publication(), None);
    }

    /// The reply layout and the destination bound are two owners answering one
    /// question, so they have to agree: what the encoder wrote is what the
    /// window says was written.
    #[test]
    fn the_encoded_reply_length_is_the_window_the_query_reports() {
        let query = pending(4096);
        let mut out = vec![0u8; query.capacity() as usize];
        let written = info_reply::encode(
            ReplyBounds {
                key_table_len: 5,
                count: 8,
            },
            &[(1, 1024), (3, 32), (4, 0)],
            &mut out,
        );
        assert_eq!(written.dropped, 0);
        let done = query.answer(written.bytes as u64).expect("fits");
        assert_eq!(
            done.write().expect("answered").bytes.length,
            written.bytes as u64
        );
    }

    #[test]
    fn every_stall_has_its_own_slug() {
        assert_ne!(
            Stall::NoAnswer.slug(),
            Stall::ReplyTooLarge {
                needed: 0,
                available: 0
            }
            .slug()
        );
    }
}
