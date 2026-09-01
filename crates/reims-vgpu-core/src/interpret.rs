//! The serial reference interpreter: what the guest is owed, executed one
//! transaction at a time.
//!
//! # Why a second executor, and why a slow one
//!
//! The replacement's whole risk is that a parallel schedule and a serial one
//! stop meaning the same thing. That is only checkable against something that
//! *defines* the serial meaning, so this is it: a backend-independent
//! interpreter that runs transactions in ingress order, one at a time, with no
//! concurrency to be wrong about.
//!
//! It is not a fallback path and it will never execute a guest's frame. What it
//! produces is a **trace** — an ordered list of the things a guest could
//! observe — and the seam that introduces parallel scheduling has to produce the
//! same trace or explain why not.
//!
//! # What counts as observable
//!
//! [`Observation`] is deliberately short. A guest cannot see how many host
//! submissions happened, which queue ran the work, or whether a transfer was a
//! copy or an import. It can see a completion stamp reach a value, a content
//! version become current, an event advance, and a refusal on the failure
//! channel. Anything not on that list is an implementation detail, and putting
//! it on the list would make the equivalence test fail for changes that are not
//! guest-visible.
//!
//! # Publication order is the one rule the trace enforces
//!
//! A transaction's content versions become visible when its work completes.
//! Its completion stamp becomes visible later, when ordered guest publication
//! releases it — see [`crate::publish`] — and never earlier. So the two halves
//! are [`Interpreter::complete`], which applies the work and hands back the
//! stamp the transaction now *owes*, and [`Interpreter::publish`], which pays
//! it. [`Interpreter::run`] is the two back to back, which is what a schedule
//! of one transaction at a time makes them.
//!
//! Keeping them apart is the point: a guest that polled the stamp and then read
//! the content must not be able to see the flag without the bytes. An
//! interpreter that wrote the stamp inside `complete` would agree with an
//! implementation that has exactly that bug.

use crate::access::{AccessKey, ContentVersion};
use crate::content::{ContentLedger, Replica};
use crate::exec::{ExecTransaction, Prerequisite, ResolvedOperation};
use crate::identity::{
    CompletionStamp, IngressOrdinal, ResourceId, SessionGeneration, StampSlot, StampValue,
};
use crate::sync::{EventKind, FenceKind};
use std::collections::HashMap;

/// Something a guest could observe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observation {
    /// A region of a backing's content reached a version.
    ///
    /// The region is part of the observation, not decoration. A guest that
    /// wrote two disjoint ranges of one buffer observes two independent
    /// histories, and folding them into one per-backing history would make two
    /// legal orders read as a disagreement.
    VersionPublished {
        backing: crate::access::BackingId,
        region: AccessKey,
        version: ContentVersion,
    },
    /// A completion stamp reached a value.
    StampPublished { slot: StampSlot, value: StampValue },
    /// An event's monotonic generation advanced.
    EventAdvanced { event: ResourceId, to: u64 },
    /// A fence was updated by the encoder that owns it.
    FenceUpdated { fence: ResourceId },
    /// A published version reached memory something newer already held, so
    /// none of it — or only part of it — became current.
    ///
    /// A completion, not a refusal: the transaction ran and its work happened.
    /// What did not happen is the bytes becoming readable, because a newer
    /// write owns them. That is a lawful outcome of two writers racing and it
    /// is exactly the outcome that must not be silent — see
    /// [`crate::coverage`]. `landed` is what did become current, in bytes.
    VersionBeaten {
        backing: crate::access::BackingId,
        region: AccessKey,
        version: ContentVersion,
        landed: u64,
    },
    /// A transaction could not run, with the reason.
    Refused {
        ingress: IngressOrdinal,
        reason: Refusal,
    },
}

/// Why the interpreter would not run a transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A prerequisite is not satisfied and, in a serial schedule, never will
    /// be: everything that could satisfy it has already run.
    ///
    /// This is the serial interpreter's version of the diagnosis
    /// [`crate::ready::Scheduler::stalled`] makes for a concurrent one, and it
    /// is a *stronger* statement here — running one at a time means nothing is
    /// outstanding, so an unmet wait is unmeetable.
    UnmeetableWait,
    /// The transaction belongs to a generation that has closed.
    StaleGeneration,
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::UnmeetableWait => "interpret_unmeetable_wait",
            Self::StaleGeneration => "interpret_stale_generation",
        }
    }
}

/// What running one transaction did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Ran,
    Refused(Refusal),
}

/// The serial reference interpreter.
///
/// Holds the semantic state a guest can observe and nothing else. No queue, no
/// submission, no host object — the point is that anything it cannot represent
/// is, by construction, not guest-visible.
#[derive(Debug)]
pub struct Interpreter {
    /// The semantic lifetime work must belong to.
    ///
    /// A reset opens a new one. Work from a closed generation is refused rather
    /// than run: it names objects that no longer exist, and running it would be
    /// executing against whatever now occupies their slots.
    generation: SessionGeneration,
    content: ContentLedger,
    events: HashMap<ResourceId, u64>,
    fences: HashMap<ResourceId, u64>,
    stamps: HashMap<StampSlot, StampValue>,
    trace: Vec<Observation>,
    ran: usize,
    refused: usize,
}

impl Default for Interpreter {
    fn default() -> Self {
        Self {
            generation: SessionGeneration::FIRST,
            content: ContentLedger::default(),
            events: HashMap::new(),
            fences: HashMap::new(),
            stamps: HashMap::new(),
            trace: Vec::new(),
            ran: 0,
            refused: 0,
        }
    }
}

impl Interpreter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The generation work must belong to.
    #[must_use]
    pub const fn generation(&self) -> SessionGeneration {
        self.generation
    }

    /// Open the next generation, as a reset does.
    ///
    /// Everything the previous generation published stays published — a stamp a
    /// guest already read does not un-read — and everything it had outstanding
    /// is simply never run, because a serial interpreter has nothing
    /// outstanding. What changes is which work is admissible from here on.
    pub fn reset(&mut self) {
        self.generation = self.generation.next();
    }

    /// The observation trace, in order.
    #[must_use]
    pub fn trace(&self) -> &[Observation] {
        &self.trace
    }

    /// The content ledger, for a caller declaring backings before a run.
    pub fn content_mut(&mut self) -> &mut ContentLedger {
        &mut self.content
    }

    #[must_use]
    pub fn event_generation(&self, event: ResourceId) -> u64 {
        self.events.get(&event).copied().unwrap_or(0)
    }

    #[must_use]
    pub fn stamp(&self, slot: StampSlot) -> Option<StampValue> {
        self.stamps.get(&slot).copied()
    }

    /// How many transactions ran and how many were refused.
    #[must_use]
    pub const fn census(&self) -> (usize, usize) {
        (self.ran, self.refused)
    }

    /// Run one transaction to completion and publish its stamp at once.
    ///
    /// Serial in the strongest sense: when this returns, the transaction's work
    /// is done and everything it publishes is visible. There is no pending
    /// state for a later call to discover.
    ///
    /// That is only the serial case. A device publishes in channel order, so
    /// the two halves are [`Self::complete`] and [`Self::publish`]; this is
    /// them back to back, which is what a schedule of one transaction at a
    /// time makes them.
    pub fn run(&mut self, tx: &ExecTransaction) -> Outcome {
        match self.complete(tx) {
            Ok(owed) => {
                if let Some(stamp) = owed {
                    self.publish(stamp);
                }
                Outcome::Ran
            }
            Err(refusal) => Outcome::Refused(refusal),
        }
    }

    /// Apply a transaction's work and publish everything that becomes visible
    /// when it completes: its records' effects and its content versions.
    ///
    /// Returns the completion stamp it now **owes**. A stamp is not visible at
    /// completion — see [`crate::publish`] — so handing it back rather than
    /// writing it is what keeps the two events apart in the reference as well
    /// as in the device.
    ///
    /// # Errors
    ///
    /// The refusal, which is also appended to the trace: a guest that is
    /// refused observes the refusal.
    pub fn complete(&mut self, tx: &ExecTransaction) -> Result<Option<CompletionStamp>, Refusal> {
        if tx.session != self.generation {
            self.refuse(tx.ingress, Refusal::StaleGeneration);
            return Err(Refusal::StaleGeneration);
        }
        if let Some(refusal) = self.unmet_prerequisite(tx) {
            self.refuse(tx.ingress, refusal);
            return Err(refusal);
        }

        // The records run in order, and the ones with observable state are the
        // synchronisation records. Everything else contributes through the
        // transaction's accesses, which is where its memory effect lives.
        for record in tx.records() {
            self.apply_record(&record.op);
        }

        // The bytes land, and then the version that says they are current
        // becomes visible. Both come from the same list — the accesses — so a
        // version cannot be published for memory nothing wrote.
        for published in tx.published_versions() {
            // The version the access reserved, not one this ledger mints: the
            // reservation happened when the transaction was planned, and a
            // completion that took a fresh number here would beat every writer
            // that reserved after it and lose to none.
            let beaten = written_bytes(&self.content, published.region).map(|bytes| {
                self.content.materialize(
                    published.backing,
                    bytes,
                    published.to,
                    Replica::DeviceOwned,
                )
            });
            match beaten {
                Some(applied) if applied.was_partly_stale() => {
                    self.trace.push(Observation::VersionBeaten {
                        backing: published.backing,
                        region: published.region,
                        version: published.to,
                        landed: applied.taken.len(),
                    });
                    if applied.is_empty() {
                        // Nothing became current, so nothing was published.
                        continue;
                    }
                }
                // A subresource or whole-backing write names no bytes this
                // crate can place — see `written_bytes` — so its effect is the
                // version alone and there is nothing for a newer write to have
                // beaten it over.
                _ => {}
            }
            self.trace.push(Observation::VersionPublished {
                backing: published.backing,
                region: published.region,
                version: published.to,
            });
        }
        self.ran += 1;
        Ok(tx.publication.stamp)
    }

    /// Make a completion word readable, as ordered guest publication releases
    /// it.
    ///
    /// Later in the wrapping order, and only observed when it advances. A
    /// completion word is a monotone point the guest polls: a value that does
    /// not advance it writes nothing the guest can read, and a plain overwrite
    /// would let the slot go backwards. This is the same rule
    /// [`crate::ready::Scheduler::publish`] applies and the same rule a signal
    /// that does not advance an event gets above — stating it in one of the
    /// three places and not the others is how the reference and the scheduler
    /// come to mean different things.
    pub fn publish(&mut self, stamp: CompletionStamp) {
        let standing = self.stamps.get(&stamp.slot).copied();
        if standing.is_none_or(|at| stamp.value.follows(at)) {
            self.stamps.insert(stamp.slot, stamp.value);
            self.trace.push(Observation::StampPublished {
                slot: stamp.slot,
                value: stamp.value,
            });
        }
    }

    fn refuse(&mut self, ingress: IngressOrdinal, reason: Refusal) {
        self.refused += 1;
        self.trace.push(Observation::Refused { ingress, reason });
    }

    /// The first prerequisite this transaction cannot meet, if any.
    fn unmet_prerequisite(&self, tx: &ExecTransaction) -> Option<Refusal> {
        for prerequisite in &tx.prerequisites {
            let met = match *prerequisite {
                Prerequisite::Stamp(wait) => self
                    .stamps
                    .get(&wait.slot)
                    .is_some_and(|published| !wait.value.follows(*published)),
                Prerequisite::Event { event, value } => self.event_generation(event) >= value,
                // A fence is encoder-scoped and its producer is inside this
                // packet or an earlier one; a serial run has already finished
                // every earlier one, so an outstanding fence is one this packet
                // updates itself.
                Prerequisite::Fence { fence } => self.fences.contains_key(&fence),
            };
            if !met {
                return Some(Refusal::UnmeetableWait);
            }
        }
        None
    }

    fn apply_record(&mut self, op: &ResolvedOperation) {
        match op {
            ResolvedOperation::Event(event) => match event.kind {
                EventKind::Signal => {
                    let current = self.event_generation(event.event);
                    if event.advances(current) {
                        self.events.insert(event.event, event.value);
                        self.trace.push(Observation::EventAdvanced {
                            event: event.event,
                            to: event.value,
                        });
                    }
                }
                // A wait inside a serial run is satisfied or the transaction
                // would not have started; the prerequisite check is where an
                // unmeetable one is caught.
                EventKind::Wait => {}
            },
            ResolvedOperation::Fence(fence) => match fence.kind {
                FenceKind::Update => {
                    *self.fences.entry(fence.fence).or_insert(0) += 1;
                    self.trace
                        .push(Observation::FenceUpdated { fence: fence.fence });
                }
                FenceKind::Wait => {}
            },
            // Everything else contributes through the transaction's accesses.
            // A barrier orders nothing in a schedule that is already serial, a
            // content directive is answered by whatever placement an executor
            // chose, and a draw's memory effect is its declared participation.
            ResolvedOperation::EncoderBoundary(_)
            | ResolvedOperation::Render(_)
            | ResolvedOperation::Compute(_)
            | ResolvedOperation::Blit(_)
            | ResolvedOperation::Barrier(_)
            | ResolvedOperation::ResourceState(_)
            | ResolvedOperation::IndirectCommand(_) => {}
        }
    }
}

/// The byte range a write access covers, when it names one.
///
/// Three answers, and the two `None`s are not the same fact.
///
/// A `Range` names its bytes. `Whole` names the backing, and the backing's
/// bytes are the extent its declaration gave it — so this asks the ledger
/// rather than returning nothing. It used to return nothing, and a
/// whole-backing write therefore published a version over no bytes: nothing
/// was covered, so a later *older* write was not beaten by it, and the replica
/// that produced the content did not become fresh for it — which makes the
/// next read from that replica owe a transfer that copies stale bytes over
/// what the device just wrote.
///
/// A subresource write is the genuine `None`: it names image coordinates
/// rather than bytes, and relating the two needs the image's layout, which is
/// an executor's and not this crate's. Its effect is the version alone. So is
/// a heap declaration's and an unparticipating access's, neither of which
/// names memory at all.
fn written_bytes(
    content: &crate::content::ContentLedger,
    key: AccessKey,
) -> Option<crate::access::ByteRange> {
    match key {
        AccessKey::Range(_, range) => Some(range),
        AccessKey::Whole(resource) => content.extent(resource.backing),
        AccessKey::Subresource(..) | AccessKey::Heap(_) | AccessKey::DomainOnly => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::{AccessIntent, AccessMode, BackingId, ByteRange, ResourceKey};
    use crate::exec::ExecBuilder;
    use crate::identity::{
        ChannelId, ChannelSequence, CompletionStamp, ObjectListRef, SlotGeneration, StampWait,
    };
    use crate::stream::SegmentKind;
    use crate::sync::EventOp;

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn builder(ingress: u64) -> ExecBuilder {
        ExecBuilder::new(
            SessionGeneration::FIRST,
            ChannelId(1),
            ChannelSequence(ingress),
            IngressOrdinal(ingress),
        )
    }

    fn signal(event: ResourceId, value: u64) -> ResolvedOperation {
        ResolvedOperation::Event(EventOp {
            kind: EventKind::Signal,
            event,
            value,
        })
    }

    fn write_access(backing: u64, offset: u64, length: u64) -> AccessIntent {
        AccessIntent {
            domain: ChannelId(1),
            key: AccessKey::Range(
                ResourceKey {
                    backing: BackingId(backing),
                    heap: None,
                },
                ByteRange { offset, length },
            ),
            mode: AccessMode::Write,
            api_stages: 0,
            input_content_version: None,
            output_content_version: Some(ContentVersion(1)),
        }
    }

    /// A completion word is a monotone point, so a stamp that does not advance
    /// it is not something a guest polling that word can observe — and a plain
    /// overwrite would let the slot go backwards, which is the failure a guest
    /// waiting on the higher value never wakes from.
    #[test]
    fn a_stamp_that_does_not_advance_its_slot_publishes_nothing() {
        let mut interp = Interpreter::new();
        for value in [9u32, 4, 9, 10] {
            let mut b = builder(u64::from(value));
            b.publish_stamp(CompletionStamp {
                slot: StampSlot(3),
                value: StampValue(value),
            });
            assert_eq!(interp.run(&b.finish().expect("frozen")), Outcome::Ran);
        }
        assert_eq!(
            interp.trace(),
            &[
                Observation::StampPublished {
                    slot: StampSlot(3),
                    value: StampValue(9)
                },
                Observation::StampPublished {
                    slot: StampSlot(3),
                    value: StampValue(10)
                },
            ],
            "4 is behind 9 and the second 9 is 9; neither is a new reading"
        );
        assert_eq!(interp.stamp(StampSlot(3)), Some(StampValue(10)));
    }

    /// And the wrapping order is the one that decides, so a timeline that
    /// wraps keeps advancing rather than freezing at `u32::MAX`.
    #[test]
    fn a_wrapped_stamp_still_advances_its_slot() {
        let mut interp = Interpreter::new();
        for value in [u32::MAX - 1, u32::MAX, 0, 1] {
            let mut b = builder(u64::from(value) + 1);
            b.publish_stamp(CompletionStamp {
                slot: StampSlot(3),
                value: StampValue(value),
            });
            assert_eq!(interp.run(&b.finish().expect("frozen")), Outcome::Ran);
        }
        assert_eq!(interp.trace().len(), 4, "every step advanced");
        assert_eq!(interp.stamp(StampSlot(3)), Some(StampValue(1)));
    }

    /// The publication order is versions then stamp, and it is the rule the
    /// trace exists to be able to fail on.
    #[test]
    fn a_transaction_publishes_its_versions_before_its_stamp() {
        let mut b = builder(1);
        let access = AccessIntent {
            output_content_version: Some(ContentVersion(2)),
            ..write_access(7, 0, 64)
        };
        b.declare_access(access);
        b.publish_stamp(CompletionStamp {
            slot: StampSlot(3),
            value: StampValue(9),
        });
        let tx = b.finish().expect("frozen");

        let mut interp = Interpreter::new();
        assert_eq!(interp.run(&tx), Outcome::Ran);
        assert_eq!(
            interp.trace(),
            &[
                Observation::VersionPublished {
                    backing: BackingId(7),
                    region: access.key,
                    version: ContentVersion(2)
                },
                Observation::StampPublished {
                    slot: StampSlot(3),
                    value: StampValue(9)
                },
            ]
        );
        assert_eq!(interp.stamp(StampSlot(3)), Some(StampValue(9)));
    }

    /// A wait for a stamp an earlier transaction published is met; one for a
    /// stamp nothing published is unmeetable, because in a serial run there is
    /// nothing outstanding to meet it later.
    #[test]
    fn an_unmet_wait_in_a_serial_run_is_unmeetable_rather_than_pending() {
        let mut producer = builder(1);
        producer.publish_stamp(CompletionStamp {
            slot: StampSlot(1),
            value: StampValue(5),
        });
        let producer = producer.finish().expect("frozen");

        let mut waiter = builder(2);
        waiter.require(Prerequisite::Stamp(StampWait {
            slot: StampSlot(1),
            value: StampValue(5),
        }));
        let waiter = waiter.finish().expect("frozen");

        let mut early = Interpreter::new();
        assert_eq!(
            early.run(&waiter),
            Outcome::Refused(Refusal::UnmeetableWait),
            "nothing has published slot 1"
        );

        let mut ordered = Interpreter::new();
        assert_eq!(ordered.run(&producer), Outcome::Ran);
        assert_eq!(ordered.run(&waiter), Outcome::Ran);
        assert_eq!(ordered.census(), (2, 0));
    }

    /// A signal that advances the generation is observable; one that does not
    /// is silent, by the API's own monotonic rule.
    #[test]
    fn only_an_advancing_signal_reaches_the_trace() {
        let mut b = builder(1);
        b.begin_segment(SegmentKind::Event.wire_type(), false)
            .expect("open");
        b.record(signal(res(4), 5)).expect("record");
        b.record(signal(res(4), 3)).expect("record");
        b.record(signal(res(4), 7)).expect("record");
        b.end_segment().expect("end");
        let tx = b.finish().expect("frozen");

        let mut interp = Interpreter::new();
        interp.run(&tx);
        assert_eq!(
            interp.trace(),
            &[
                Observation::EventAdvanced {
                    event: res(4),
                    to: 5
                },
                Observation::EventAdvanced {
                    event: res(4),
                    to: 7
                },
            ]
        );
        assert_eq!(interp.event_generation(res(4)), 7);
    }

    /// An event wait is met by a value an earlier transaction signalled, and
    /// waiting for a value at or below the generation is met immediately.
    #[test]
    fn an_event_wait_is_met_at_or_past_its_value() {
        let mut producer = builder(1);
        producer
            .begin_segment(SegmentKind::Event.wire_type(), false)
            .expect("open");
        producer.record(signal(res(4), 5)).expect("record");
        producer.end_segment().expect("end");
        let producer = producer.finish().expect("frozen");

        let mut interp = Interpreter::new();
        interp.run(&producer);

        for (value, expected) in [(4u64, Outcome::Ran), (5, Outcome::Ran)] {
            let mut b = builder(2);
            b.require(Prerequisite::Event {
                event: res(4),
                value,
            });
            assert_eq!(interp.run(&b.finish().expect("frozen")), expected);
        }
        let mut b = builder(3);
        b.require(Prerequisite::Event {
            event: res(4),
            value: 6,
        });
        assert_eq!(
            interp.run(&b.finish().expect("frozen")),
            Outcome::Refused(Refusal::UnmeetableWait)
        );
    }

    /// A byte-ranged write reaches the content ledger, and each other access
    /// shape names its own bytes or says why it cannot.
    ///
    /// The whole backing is the extent its declaration gave it; a backing no
    /// declaration reached names nothing, because the model does not know how
    /// big it is; and a subresource names nothing because relating image
    /// coordinates to bytes needs a layout this crate cannot see. Three
    /// answers, and the two silences are different facts.
    #[test]
    fn a_ranged_write_advances_content_and_each_shape_names_its_own_bytes() {
        let mut b = builder(1);
        b.declare_access(write_access(9, 0, 0x40));
        let tx = b.finish().expect("frozen");

        let mut interp = Interpreter::new();
        interp.content_mut().declare(
            BackingId(9),
            ByteRange {
                offset: 0,
                length: 0x100,
            },
            Replica::GuestPages,
        );
        interp.run(&tx);
        assert!(interp.content.is_fresh(
            BackingId(9),
            ByteRange {
                offset: 0,
                length: 0x40
            },
            Replica::DeviceOwned
        ));
        assert!(!interp.content.is_fresh(
            BackingId(9),
            ByteRange {
                offset: 0,
                length: 0x40
            },
            Replica::GuestPages
        ));

        // A whole-backing write names the extent the declaration gave it.
        // Above it named nothing, so a whole-backing write covered no bytes and
        // published a version over memory it never claimed.
        let whole = AccessKey::Whole(ResourceKey {
            backing: BackingId(9),
            heap: None,
        });
        assert_eq!(
            written_bytes(&interp.content, whole),
            Some(ByteRange {
                offset: 0,
                length: 0x100
            })
        );
        // And a backing no declaration reached still names nothing: the model
        // does not know how big it is, and a guessed size would claim memory
        // the guest never gave it.
        assert_eq!(
            written_bytes(
                &interp.content,
                AccessKey::Whole(ResourceKey {
                    backing: BackingId(404),
                    heap: None
                })
            ),
            None
        );
        // A subresource is the genuine unknown: its bytes need a layout.
        assert_eq!(
            written_bytes(
                &interp.content,
                AccessKey::Subresource(
                    ResourceKey {
                        backing: BackingId(9),
                        heap: None
                    },
                    crate::access::SubresourceRange {
                        base_level: 0,
                        level_count: 1,
                        base_slice: 0,
                        slice_count: 1,
                        plane: 0,
                    }
                )
            ),
            None
        );
    }

    /// A refusal is observable, so a trace comparison catches a schedule that
    /// silently accepted what the serial one refused.
    #[test]
    fn a_refusal_is_part_of_the_trace() {
        let mut b = builder(1);
        b.require(Prerequisite::Stamp(StampWait {
            slot: StampSlot(1),
            value: StampValue(1),
        }));
        let tx = b.finish().expect("frozen");
        let mut interp = Interpreter::new();
        interp.run(&tx);
        assert_eq!(
            interp.trace(),
            &[Observation::Refused {
                ingress: IngressOrdinal(1),
                reason: Refusal::UnmeetableWait,
            }]
        );
        assert_eq!(interp.census(), (0, 1));
    }

    /// Work from a closed generation is refused, and refused by name rather
    /// than by silently doing nothing.
    #[test]
    fn work_from_a_closed_generation_is_refused() {
        let tx = builder(1).finish().expect("frozen");
        let mut interp = Interpreter::new();
        interp.reset();
        assert_eq!(
            interp.run(&tx),
            Outcome::Refused(Refusal::StaleGeneration),
            "the transaction was built in the first generation"
        );
        assert_eq!(
            interp.trace(),
            &[Observation::Refused {
                ingress: IngressOrdinal(1),
                reason: Refusal::StaleGeneration,
            }]
        );
    }

    /// A reset does not un-publish what the previous generation published.
    #[test]
    fn a_reset_keeps_what_was_already_visible() {
        let mut b = builder(1);
        b.publish_stamp(CompletionStamp {
            slot: StampSlot(2),
            value: StampValue(4),
        });
        let tx = b.finish().expect("frozen");
        let mut interp = Interpreter::new();
        interp.run(&tx);
        interp.reset();
        assert_eq!(interp.stamp(StampSlot(2)), Some(StampValue(4)));
        assert_ne!(interp.generation(), SessionGeneration::FIRST);
    }

    /// Running the same transactions twice produces the same trace. The
    /// interpreter is the definition of the serial meaning, so it must not
    /// depend on anything but its inputs.
    #[test]
    fn the_trace_is_a_function_of_the_transactions_alone() {
        let make = || {
            let mut b = builder(1);
            b.begin_segment(SegmentKind::Event.wire_type(), false)
                .expect("open");
            b.record(signal(res(4), 2)).expect("record");
            b.end_segment().expect("end");
            b.publish_stamp(CompletionStamp {
                slot: StampSlot(1),
                value: StampValue(1),
            });
            b.finish().expect("frozen")
        };
        let tx = make();
        let mut a = Interpreter::new();
        let mut b = Interpreter::new();
        a.run(&tx);
        b.run(&make());
        assert_eq!(a.trace(), b.trace());
    }

    /// Two transactions writing disjoint ranges of one backing are both
    /// current, whichever order they complete in.
    ///
    /// The failure a per-backing version cannot avoid: the later reservation is
    /// the higher number, so under one version per backing the earlier writer's
    /// completion arrives holding a version the backing has already passed.
    /// Here they never meet, because they cover different bytes — and the
    /// interpreter is the reference every parallel schedule is checked against,
    /// so getting this wrong here would make the whole equivalence proof agree
    /// about the wrong answer.
    #[test]
    fn two_writers_of_disjoint_ranges_are_both_current_in_either_order() {
        for late_first in [false, true] {
            let mut interp = Interpreter::new();
            interp.content_mut().declare(
                BackingId(7),
                ByteRange {
                    offset: 0,
                    length: 128,
                },
                Replica::GuestPages,
            );
            let front = AccessIntent {
                output_content_version: Some(ContentVersion(5)),
                ..write_access(7, 0, 64)
            };
            let back = AccessIntent {
                output_content_version: Some(ContentVersion(6)),
                ..write_access(7, 64, 64)
            };
            let order = if late_first {
                [back, front]
            } else {
                [front, back]
            };
            for (n, access) in order.into_iter().enumerate() {
                let mut b = builder(n as u64 + 1);
                b.declare_access(access);
                assert_eq!(interp.run(&b.finish().expect("frozen")), Outcome::Ran);
            }
            assert!(
                !interp
                    .trace()
                    .iter()
                    .any(|o| matches!(o, Observation::VersionBeaten { .. })),
                "disjoint writers must not beat each other ({late_first})"
            );
            let content = interp.content_mut();
            assert_eq!(
                content.version_of(
                    BackingId(7),
                    ByteRange {
                        offset: 0,
                        length: 64
                    }
                ),
                Some(ContentVersion(5))
            );
            assert_eq!(
                content.version_of(
                    BackingId(7),
                    ByteRange {
                        offset: 64,
                        length: 64
                    }
                ),
                Some(ContentVersion(6))
            );
        }
    }

    /// A completion that lost the race publishes nothing and says so.
    ///
    /// The stale-completion rule, at the seam where a guest would see it: the
    /// newer write owns the bytes, so the older one's are never readable. It is
    /// an observation and not a refusal — the transaction ran, and what did not
    /// happen is its bytes becoming visible.
    #[test]
    fn a_completion_beaten_by_newer_content_publishes_nothing_and_names_it() {
        let mut interp = Interpreter::new();
        let newer = AccessIntent {
            output_content_version: Some(ContentVersion(9)),
            ..write_access(7, 0, 128)
        };
        let older = AccessIntent {
            output_content_version: Some(ContentVersion(4)),
            ..write_access(7, 32, 64)
        };
        for (n, access) in [newer, older].into_iter().enumerate() {
            let mut b = builder(n as u64 + 1);
            b.declare_access(access);
            assert_eq!(interp.run(&b.finish().expect("frozen")), Outcome::Ran);
        }
        assert_eq!(
            interp.trace(),
            &[
                Observation::VersionPublished {
                    backing: BackingId(7),
                    region: newer.key,
                    version: ContentVersion(9),
                },
                Observation::VersionBeaten {
                    backing: BackingId(7),
                    region: older.key,
                    version: ContentVersion(4),
                    landed: 0,
                },
            ],
            "the beaten write must not also read as published"
        );
        assert_eq!(
            interp.content_mut().version_of(
                BackingId(7),
                ByteRange {
                    offset: 32,
                    length: 64
                }
            ),
            Some(ContentVersion(9)),
            "the newer content still owns those bytes"
        );
    }

    /// A partly beaten completion publishes: some of its bytes did become
    /// current, and a guest reading those sees them.
    #[test]
    fn a_partly_beaten_completion_publishes_the_part_that_landed() {
        let mut interp = Interpreter::new();
        let newer = AccessIntent {
            output_content_version: Some(ContentVersion(9)),
            ..write_access(7, 64, 64)
        };
        let straddling = AccessIntent {
            output_content_version: Some(ContentVersion(4)),
            ..write_access(7, 0, 128)
        };
        for (n, access) in [newer, straddling].into_iter().enumerate() {
            let mut b = builder(n as u64 + 1);
            b.declare_access(access);
            assert_eq!(interp.run(&b.finish().expect("frozen")), Outcome::Ran);
        }
        assert_eq!(
            interp.trace()[1..],
            [
                Observation::VersionBeaten {
                    backing: BackingId(7),
                    region: straddling.key,
                    version: ContentVersion(4),
                    landed: 64,
                },
                Observation::VersionPublished {
                    backing: BackingId(7),
                    region: straddling.key,
                    version: ContentVersion(4),
                },
            ],
            "half landed, so it is both beaten and published"
        );
    }

    /// A whole-backing write makes the writing replica fresh for the whole
    /// backing, so a later read from it owes no transfer.
    ///
    /// The failure this replaces: a `Whole` access named no bytes, so the
    /// device's write covered nothing and the guest stayed fresh for
    /// everything. The next read from device storage then owed a copy *from*
    /// the guest — stale bytes, over content the device had just produced,
    /// with a version published saying the device's content was current.
    #[test]
    fn a_whole_backing_write_makes_its_replica_fresh_for_the_whole_backing() {
        let extent = ByteRange {
            offset: 0,
            length: 0x100,
        };
        let mut interp = Interpreter::new();
        interp
            .content_mut()
            .declare(BackingId(9), extent, Replica::GuestPages);

        let mut b = builder(1);
        b.declare_access(AccessIntent {
            key: AccessKey::Whole(ResourceKey {
                backing: BackingId(9),
                heap: None,
            }),
            output_content_version: Some(ContentVersion(5)),
            ..write_access(9, 0, 0x100)
        });
        assert_eq!(interp.run(&b.finish().expect("frozen")), Outcome::Ran);

        let content = interp.content_mut();
        assert!(content.is_fresh(BackingId(9), extent, Replica::DeviceOwned));
        assert!(!content.is_fresh(BackingId(9), extent, Replica::GuestPages));
        assert!(
            content
                .transfer_for_read(BackingId(9), extent, Replica::DeviceOwned)
                .is_none(),
            "the device produced these bytes; nothing may be copied over them"
        );
        assert_eq!(
            content.version_of(BackingId(9), extent),
            Some(ContentVersion(5))
        );

        // And it beats an older write that overlaps it, which a version over
        // no bytes could not have done.
        let mut b = builder(2);
        b.declare_access(AccessIntent {
            output_content_version: Some(ContentVersion(2)),
            ..write_access(9, 0, 0x40)
        });
        assert_eq!(interp.run(&b.finish().expect("frozen")), Outcome::Ran);
        assert!(interp
            .trace()
            .iter()
            .any(|o| matches!(o, Observation::VersionBeaten { landed: 0, .. })));
    }
}
