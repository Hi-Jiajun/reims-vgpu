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
//! A transaction's content versions and its completion stamp become visible
//! together, and both only after the work is done. The interpreter emits the
//! versions before the stamp within one transaction, which is the order the
//! plan requires — a guest that polled the stamp and then read the content must
//! not be able to see the flag without the bytes. Emitting them in the other
//! order would make the trace agree with an implementation that has that bug.

use crate::access::{AccessKey, ContentVersion};
use crate::content::{ContentLedger, Replica};
use crate::exec::{ExecTransaction, Prerequisite, ResolvedOperation};
use crate::identity::{IngressOrdinal, ResourceId, SessionGeneration, StampSlot, StampValue};
use crate::sync::{EventKind, FenceKind};
use std::collections::HashMap;

/// Something a guest could observe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observation {
    /// A backing's content reached a version.
    VersionPublished {
        backing: crate::access::BackingId,
        version: ContentVersion,
    },
    /// A completion stamp reached a value.
    StampPublished { slot: StampSlot, value: StampValue },
    /// An event's monotonic generation advanced.
    EventAdvanced { event: ResourceId, to: u64 },
    /// A fence was updated by the encoder that owns it.
    FenceUpdated { fence: ResourceId },
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

    /// Run one transaction to completion.
    ///
    /// Serial in the strongest sense: when this returns, the transaction's work
    /// is done and everything it publishes is visible. There is no pending
    /// state for a later call to discover.
    pub fn run(&mut self, tx: &ExecTransaction) -> Outcome {
        if tx.session != self.generation {
            return self.refuse(tx.ingress, Refusal::StaleGeneration);
        }
        if let Some(refusal) = self.unmet_prerequisite(tx) {
            return self.refuse(tx.ingress, refusal);
        }

        // The records run in order, and the ones with observable state are the
        // synchronisation records. Everything else contributes through the
        // transaction's accesses, which is where its memory effect lives.
        for record in tx.records() {
            self.apply_record(&record.op);
        }

        // Writes land before anything is published, which is what makes the
        // publication order below meaningful.
        for access in &tx.accesses {
            if access.output_content_version.is_none() {
                continue;
            }
            let Some(bytes) = written_bytes(access.key) else {
                continue;
            };
            let backing = match access.key {
                AccessKey::Range(r, _) | AccessKey::Subresource(r, _) | AccessKey::Whole(r) => {
                    r.backing
                }
                AccessKey::Heap(_) | AccessKey::DomainOnly => continue,
            };
            self.content.write(backing, bytes, Replica::DeviceOwned);
        }

        // Versions first, then the stamp. See the module documentation: the
        // opposite order is the bug this trace has to be able to fail on.
        for reservation in &tx.publication.versions {
            self.trace.push(Observation::VersionPublished {
                backing: reservation.backing,
                version: reservation.to,
            });
        }
        if let Some(stamp) = tx.publication.stamp {
            self.stamps.insert(stamp.slot, stamp.value);
            self.trace.push(Observation::StampPublished {
                slot: stamp.slot,
                value: stamp.value,
            });
        }
        self.ran += 1;
        Outcome::Ran
    }

    fn refuse(&mut self, ingress: IngressOrdinal, reason: Refusal) -> Outcome {
        self.refused += 1;
        self.trace.push(Observation::Refused { ingress, reason });
        Outcome::Refused(reason)
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
/// A subresource write names image coordinates rather than bytes, and this
/// crate cannot relate the two — that needs the image's layout, which is an
/// executor's. So a subresource write moves no bytes in the ledger and its
/// effect is the version alone. Saying that here, once, is better than a caller
/// inventing a range from a subresource.
fn written_bytes(key: AccessKey) -> Option<crate::access::ByteRange> {
    match key {
        AccessKey::Range(_, range) => Some(range),
        AccessKey::Subresource(..)
        | AccessKey::Whole(_)
        | AccessKey::Heap(_)
        | AccessKey::DomainOnly => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::{AccessIntent, AccessMode, BackingId, ByteRange, ResourceKey};
    use crate::exec::{ExecBuilder, VersionReservation};
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

    /// The publication order is versions then stamp, and it is the rule the
    /// trace exists to be able to fail on.
    #[test]
    fn a_transaction_publishes_its_versions_before_its_stamp() {
        let mut b = builder(1);
        b.reserve_version(VersionReservation {
            backing: BackingId(7),
            from: ContentVersion(1),
            to: ContentVersion(2),
        });
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

    /// A byte-ranged write reaches the content ledger; a subresource write
    /// moves no bytes there, because relating image coordinates to bytes needs
    /// a layout this crate cannot see.
    #[test]
    fn a_ranged_write_advances_content_and_a_subresource_write_does_not() {
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

        assert_eq!(
            written_bytes(AccessKey::Whole(ResourceKey {
                backing: BackingId(9),
                heap: None
            })),
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
}
