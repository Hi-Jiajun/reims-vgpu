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
use reims_vgpu_protocol::fifo::{self, DeviceInfoForm};
use reims_vgpu_protocol::info_reply::ReplyBounds;
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
    /// Which of the device-info request layouts this question's packet carries,
    /// or `None` for the questions that are not device-info.
    ///
    /// The two device-info opcodes differ only in that the newer one prepends a
    /// parse ceiling, so every offset after it moves by four. The choice of
    /// form is therefore an opcode question and nothing else, and it was made
    /// at the packet arms — one arm naming one form, another naming the other,
    /// with nothing able to compare them. Reading either request at the other's
    /// offsets takes the pair count for a page frame and writes the reply to
    /// whatever page that named.
    #[must_use]
    pub const fn device_info_form(self) -> Option<DeviceInfoForm> {
        match self {
            Self::DeviceInfoLegacy => Some(DeviceInfoForm::WithoutKeyLimit),
            Self::DeviceInfo => Some(DeviceInfoForm::WithKeyLimit),
            Self::ComputeInfo | Self::HeapTextureSizeAndAlign => None,
        }
    }

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

/// What shape of answer a question expects, with the bounds its request
/// carried.
///
/// The bounds are part of the request and not of the destination. A destination
/// says how many bytes fit; these say how many pairs the guest will consume and
/// which keys its own parser reaches, and a reply is correct only under all
/// three. Carrying the pair here rather than beside it is what stops an
/// evaluator from being handed a window and left to guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplyShape {
    /// A run of `(key, value)` pairs, laid out by
    /// [`reims_vgpu_protocol::info_reply`].
    KeyValue(ReplyBounds),
    /// A record of a fixed size, which the request does not bound and the
    /// question's own layout decides.
    ///
    /// The size is carried rather than left to the destination, because the
    /// destination's length is not it: a guest that set aside a page for a
    /// sixteen-byte answer named a window, not a reply. Without the number
    /// here, "does the answer fit" has no left-hand side and every fixed reply
    /// would be judged to fit whatever it was pointed at.
    Fixed { bytes: u64 },
}

/// How many bytes this device's answer to a question occupies.
///
/// **The values are the host's and this crate cannot see a host — the length
/// is all a serial reference needs.** What a guest observes of a reply is that
/// a window's content reached a version, not what is in it, so a reference
/// that knows how long the answer is can decide everything the ordering
/// contract turns on: whether it fits, whether a version becomes current, and
/// whether the stamp is all the guest gets.
///
/// A separate trait rather than a field on the request, for the reason
/// [`crate::resolve::RefResolver`] is one: the request is the guest's and this
/// is the device's, and a request that carried its own answer's length would be
/// a guest deciding what this device can answer.
pub trait AnswerLength: core::fmt::Debug {
    /// The bytes an answer to `request` would write, or `None` when this
    /// device has no answer for the question at all.
    fn bytes(&self, request: &QueryRequest) -> Option<u64>;
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
    pub reply: ReplyShape,
}

/// A query packet's request words, as its own layout carries them.
///
/// One variant per layout rather than per question, because the two device-info
/// opcodes share a decoder and differ by a form. Which form is
/// [`QueryKind::device_info_form`]'s answer, and [`resolve`] checks that the
/// words it was handed came from the layout the kind names — a check no caller
/// can make for itself, since the two forms decode to the same Rust type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestWords {
    DeviceInfo(fifo::DeviceInfoRequest),
    ComputeInfo(fifo::ComputeInfoRequest),
    /// The heap-texture query, whose request is a serialized texture descriptor
    /// and whose reply is a fixed record. Nothing about it bounds the reply, so
    /// there are no words to carry here.
    HeapTexture,
}

/// Why a request's words are not the ones its question asks with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveRefusal {
    /// The words came from a different layout than the kind names.
    ///
    /// The dangerous case is the two device-info forms, which decode to one
    /// Rust type: a `DeviceInfoRequest` read at the other form's offsets is a
    /// well-typed value whose count is a page frame.
    WrongLayout { kind: QueryKind },
}

impl ResolveRefusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::WrongLayout { .. } => "query_request_wrong_layout",
        }
    }
}

/// Join a question, the words its request carried and the destination its reply
/// goes to into one request.
///
/// **The destination is resolved and this function does not resolve it.** The
/// address a request names is translated by whoever owns translation; a
/// destination arriving here as an address would be one this crate could not
/// bound. What this owns is the pairing: which bounds belong to which question,
/// and that the words came from the layout the question uses.
///
/// # Errors
///
/// [`ResolveRefusal::WrongLayout`] when the words are not the kind's own.
pub fn resolve(
    kind: QueryKind,
    words: RequestWords,
    destination: ReplyDestination,
) -> Result<QueryRequest, ResolveRefusal> {
    let reply = match (kind, words) {
        (_, RequestWords::DeviceInfo(request)) if kind.device_info_form() == Some(request.form) => {
            ReplyShape::KeyValue(request.reply_bounds())
        }
        (QueryKind::ComputeInfo, RequestWords::ComputeInfo(request)) => {
            ReplyShape::KeyValue(request.reply_bounds())
        }
        (QueryKind::HeapTextureSizeAndAlign, RequestWords::HeapTexture) => ReplyShape::Fixed {
            // An `MTLSizeAndAlign`, whose length is the wire framing's and not
            // restated here.
            bytes: reims_vgpu_protocol::fifo::HEAP_TEXTURE_REPLY_LEN as u64,
        },
        _ => return Err(ResolveRefusal::WrongLayout { kind }),
    };
    Ok(QueryRequest {
        kind,
        destination,
        reply,
    })
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
                reply: ReplyShape::KeyValue(ReplyBounds {
                    key_table_len: 5,
                    count: 512,
                }),
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

    fn device_info_words(form: DeviceInfoForm) -> fifo::DeviceInfoRequest {
        let mut bytes = [0u8; 12];
        let bytes = &mut bytes[..form.request_len()];
        if let Some(at) = form.key_table_len_offset() {
            bytes[at..at + 4].copy_from_slice(&18u32.to_le_bytes());
        }
        let at = form.pair_capacity_offset();
        bytes[at..at + 4].copy_from_slice(&512u32.to_le_bytes());
        let at = form.reply_pfn_offset();
        bytes[at..at + 4].copy_from_slice(&0x40u32.to_le_bytes());
        fifo::decode_device_info(form, bytes).expect("built at the form's own length")
    }

    /// Each question is resolved with the layout its own opcode carries, and
    /// the bounds that reach the request are the ones the words held.
    #[test]
    fn a_question_is_joined_to_the_words_its_own_layout_carries() {
        let words = device_info_words(DeviceInfoForm::WithKeyLimit);
        let resolved = resolve(
            QueryKind::DeviceInfo,
            RequestWords::DeviceInfo(words),
            destination(4096),
        )
        .expect("the newer opcode's form");
        assert_eq!(
            resolved.reply,
            ReplyShape::KeyValue(ReplyBounds {
                key_table_len: 18,
                count: 512,
            })
        );

        // The older form carries no ceiling, and "the count alone bounds it" is
        // one derivation rather than a `u32::MAX` written at each caller.
        let legacy = resolve(
            QueryKind::DeviceInfoLegacy,
            RequestWords::DeviceInfo(device_info_words(DeviceInfoForm::WithoutKeyLimit)),
            destination(4096),
        )
        .expect("the older opcode's form");
        assert_eq!(
            legacy.reply,
            ReplyShape::KeyValue(ReplyBounds {
                key_table_len: u32::MAX,
                count: 512,
            })
        );
    }

    /// Both device-info forms decode to one Rust type, so handing a question
    /// the other form's words type-checks. It is refused here.
    ///
    /// This is the whole reason the join exists. The two forms' pair counts sit
    /// four bytes apart, so the wrong form's request is a well-typed value
    /// whose count is a page frame — and the reply would be written to whatever
    /// page that named.
    #[test]
    fn a_question_refuses_the_other_forms_words() {
        for (kind, wrong) in [
            (QueryKind::DeviceInfo, DeviceInfoForm::WithoutKeyLimit),
            (QueryKind::DeviceInfoLegacy, DeviceInfoForm::WithKeyLimit),
        ] {
            assert_eq!(
                resolve(
                    kind,
                    RequestWords::DeviceInfo(device_info_words(wrong)),
                    destination(4096)
                ),
                Err(ResolveRefusal::WrongLayout { kind })
            );
        }
    }

    /// A question resolves from exactly one variant of the request words, and
    /// every question has one.
    #[test]
    fn every_question_resolves_from_exactly_one_layout() {
        let every: [RequestWords; 4] = [
            RequestWords::DeviceInfo(device_info_words(DeviceInfoForm::WithoutKeyLimit)),
            RequestWords::DeviceInfo(device_info_words(DeviceInfoForm::WithKeyLimit)),
            RequestWords::ComputeInfo(
                fifo::decode_compute_info(&[0u8; fifo::COMPUTE_INFO_REQUEST_LEN])
                    .expect("a full-length request"),
            ),
            RequestWords::HeapTexture,
        ];
        for kind in [
            QueryKind::DeviceInfoLegacy,
            QueryKind::DeviceInfo,
            QueryKind::ComputeInfo,
            QueryKind::HeapTextureSizeAndAlign,
        ] {
            let accepted = every
                .iter()
                .filter(|words| resolve(kind, **words, destination(64)).is_ok())
                .count();
            assert_eq!(accepted, 1, "{} accepts {accepted} layouts", kind.name());
        }
    }

    /// The heap-texture query's reply is a fixed record its request does not
    /// bound — and its length is the record's, not the destination's.
    ///
    /// The destination here is four pages, so a reply shape that took its
    /// length from the window would say 16384. What the guest reads back is an
    /// `MTLSizeAndAlign` either way.
    #[test]
    fn the_heap_texture_querys_reply_is_its_records_length_not_its_windows() {
        let resolved = resolve(
            QueryKind::HeapTextureSizeAndAlign,
            RequestWords::HeapTexture,
            destination(16384),
        )
        .expect("its own layout");
        assert_eq!(
            resolved.reply,
            ReplyShape::Fixed {
                bytes: reims_vgpu_protocol::fifo::HEAP_TEXTURE_REPLY_LEN as u64
            }
        );
        assert_eq!(resolved.destination.bytes.length, 16384);
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
                reply: ReplyShape::Fixed { bytes: 16 },
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
