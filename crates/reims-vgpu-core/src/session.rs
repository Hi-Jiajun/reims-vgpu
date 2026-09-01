//! Ingress: the one place a packet becomes a transaction, or does not become
//! one at all.
//!
//! # Refusal happens here or it happens too late
//!
//! A packet whose contract is not established must be refused *before* it is
//! given an ingress ordinal, an ordering position and a completion obligation.
//! Once it has those, every mechanism downstream will honour them — the hazard
//! compiler will order against its accesses, the scheduler will hold work for
//! its stamp, and something will eventually have to publish a completion for
//! work that was never described. A refusal at ingress costs the guest one
//! command; a refusal after admission costs it the channel.
//!
//! So [`SessionModel::admit`] decides in one place, from the closure ledger,
//! and the refusal it returns names which of the reasons applied.
//!
//! # What a session owns and what it does not
//!
//! It owns the ordinal counters, the per-channel sequences, the hazard graph
//! and the readiness service. It owns no resources, no pipelines and no host
//! objects, and it cannot: those live behind a lease whose identity carries
//! this session's generation, and the crate they live in is not this one.

use crate::access::AccessIntent;
use crate::depend::DependencyGraph;
use crate::identity::{
    ChannelId, ChannelSequence, CompletionStamp, IngressOrdinal, SessionGeneration, SessionId,
    StampWait,
};
use crate::ready::Scheduler;
use crate::transaction::{classify, DeviceTransaction, PayloadClass};
use reims_vgpu_protocol::packets::{find, Channel};
use std::collections::BTreeMap;

/// Why a packet did not become a transaction.
///
/// Each variant is one check, never shared, so a reader can tell which one
/// refused. The slug is the name it reaches a failure channel under; this crate
/// does not own a failure channel, and a caller that has one renders these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// No dispatch table entry: the opcode names no command on this channel.
    UnknownCommand { channel: Channel, opcode: u16 },
    /// A command with no established contract. Admitting it would promise
    /// ordering and completion for work the model cannot describe.
    UnestablishedContract { channel: Channel, opcode: u16 },
    /// The packet arrived after the semantic lifetime it names was closed.
    /// Not an error in the guest: a reset races in-flight submissions, and the
    /// contract is that the closed generation stops accepting rather than that
    /// the guest stops sending.
    GenerationClosed {
        named: SessionGeneration,
        current: SessionGeneration,
    },
}

impl Refusal {
    pub const fn slug(self) -> &'static str {
        match self {
            Self::UnknownCommand { .. } => "ingress_unknown_command",
            Self::UnestablishedContract { .. } => "ingress_unestablished_contract",
            Self::GenerationClosed { .. } => "ingress_generation_closed",
        }
    }
}

/// A packet as ingress receives it, before it is a transaction.
#[derive(Clone, Debug, PartialEq)]
pub struct Packet {
    pub channel: Channel,
    /// The channel this arrived on. [`Packet::channel`] says which dispatch
    /// table; this says which ordering domain, and the root channel is one
    /// domain like any other.
    pub domain: ChannelId,
    pub opcode: u16,
    pub stamp_waits: Vec<StampWait>,
    pub completion: Option<CompletionStamp>,
    /// What the packet touches, already resolved. Resolution is the caller's:
    /// it needs the namespaces, and this is the ordering plane.
    pub accesses: Vec<AccessIntent>,
}

/// What admitting a packet produced.
#[derive(Clone, Debug, PartialEq)]
pub struct Admitted {
    pub transaction: DeviceTransaction,
    /// Earlier transactions this one must not overtake.
    pub hazard_waits: Vec<IngressOrdinal>,
    /// Whether it may begin immediately.
    pub ready: bool,
}

/// The ordering and readiness plane for one semantic lifetime.
#[derive(Debug)]
pub struct SessionModel {
    id: SessionId,
    generation: SessionGeneration,
    next_ingress: IngressOrdinal,
    channel_sequence: BTreeMap<ChannelId, ChannelSequence>,
    graph: DependencyGraph,
    scheduler: Scheduler,
    refusals: usize,
}

impl SessionModel {
    #[must_use]
    pub fn new(id: SessionId) -> Self {
        Self {
            id,
            generation: SessionGeneration::FIRST,
            next_ingress: IngressOrdinal::default().next(),
            channel_sequence: BTreeMap::new(),
            graph: DependencyGraph::new(),
            scheduler: Scheduler::new(),
            refusals: 0,
        }
    }

    #[must_use]
    pub const fn id(&self) -> SessionId {
        self.id
    }

    #[must_use]
    pub const fn generation(&self) -> SessionGeneration {
        self.generation
    }

    #[must_use]
    pub const fn refusals(&self) -> usize {
        self.refusals
    }

    /// Close this semantic lifetime and open the next.
    ///
    /// New resolution stops; accepted work is not invalidated. The ordering
    /// plane is deliberately *not* cleared: a transaction accepted in the old
    /// generation still has to complete, still publishes its stamp, and still
    /// releases whatever waited on it. Dropping it here would be a reset that
    /// can lose a completion the host is still going to deliver.
    pub fn reset(&mut self) -> SessionGeneration {
        self.generation = self.generation.next();
        self.generation
    }

    /// Turn a packet into a transaction, or refuse it.
    ///
    /// # Errors
    ///
    /// Returns the one check that refused. Nothing is mutated on a refusal —
    /// no ordinal is consumed and no sequence advances — so a refused packet
    /// leaves no gap in either order for a reader to explain.
    pub fn admit(&mut self, packet: &Packet) -> Result<Admitted, Refusal> {
        let Some(judged) = find(packet.channel, packet.opcode) else {
            self.refusals += 1;
            return Err(Refusal::UnknownCommand {
                channel: packet.channel,
                opcode: packet.opcode,
            });
        };
        let Some(payload) = classify(packet.channel, packet.opcode) else {
            debug_assert!(judged.closure.blocks_cutover());
            self.refusals += 1;
            return Err(Refusal::UnestablishedContract {
                channel: packet.channel,
                opcode: packet.opcode,
            });
        };

        let ingress = self.next_ingress;
        self.next_ingress = ingress.next();
        let sequence = self
            .channel_sequence
            .entry(packet.domain)
            .or_default()
            .next();
        self.channel_sequence.insert(packet.domain, sequence);

        let hazard_waits = self.graph.admit(ingress, &packet.accesses);
        let ready = self.scheduler.admit(
            ingress,
            &hazard_waits,
            &packet.stamp_waits,
            packet.completion,
        );
        Ok(Admitted {
            transaction: DeviceTransaction {
                session: self.generation,
                channel: packet.domain,
                channel_sequence: sequence,
                ingress,
                stamp_waits: packet.stamp_waits.clone(),
                completion: packet.completion,
                payload,
                accesses: packet.accesses.clone(),
            },
            hazard_waits,
            ready,
        })
    }

    /// Complete a transaction: publish its stamp, release its dependents, and
    /// stop its accesses creating edges.
    ///
    /// The two halves are one call because they are one fact. A completion that
    /// released dependents without retiring accesses would leave a finished
    /// transaction ordering later work forever; one that retired accesses
    /// without publishing would lose the stamp.
    pub fn complete(&mut self, ingress: IngressOrdinal) {
        self.scheduler.complete(ingress);
        self.graph.retire(ingress);
    }

    /// Transactions that have become ready since the last call.
    pub fn take_ready(&mut self) -> Vec<IngressOrdinal> {
        self.scheduler.take_ready()
    }

    #[must_use]
    pub const fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    #[must_use]
    pub const fn graph(&self) -> &DependencyGraph {
        &self.graph
    }

    /// Reclaim the space retired transactions were holding.
    pub fn compact(&mut self) {
        self.graph.compact();
    }

    /// Whether a payload class reaches an executor, for a caller deciding what
    /// to hand where.
    #[must_use]
    pub const fn executes(payload: PayloadClass) -> bool {
        crate::transaction::reaches_an_executor(payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::{AccessKey, AccessMode, BackingId, ResourceKey};
    use crate::identity::{StampSlot, StampValue};

    fn packet(opcode: u16) -> Packet {
        Packet {
            channel: Channel::Child,
            domain: ChannelId(2),
            opcode,
            stamp_waits: Vec::new(),
            completion: None,
            accesses: Vec::new(),
        }
    }

    fn whole(backing: u64, mode: AccessMode) -> AccessIntent {
        AccessIntent {
            domain: ChannelId(2),
            key: AccessKey::Whole(ResourceKey {
                backing: BackingId(backing),
                heap: None,
            }),
            mode,
            api_stages: 0,
            input_content_version: None,
            output_content_version: None,
        }
    }

    /// An EXEC and a resource delete get the same envelope, and differ only in
    /// what they carry.
    #[test]
    fn every_accepted_packet_gets_the_same_envelope() {
        let mut s = SessionModel::new(SessionId(1));
        let exec = s.admit(&packet(0x37)).expect("EXEC is accepted");
        let delete = s.admit(&packet(0x25)).expect("delete is accepted");
        assert_eq!(exec.transaction.payload, PayloadClass::Exec);
        assert_eq!(delete.transaction.payload, PayloadClass::ResourceLifecycle);
        assert_eq!(exec.transaction.ingress, IngressOrdinal(1));
        assert_eq!(delete.transaction.ingress, IngressOrdinal(2));
        assert_eq!(exec.transaction.channel_sequence, ChannelSequence(1));
        assert_eq!(delete.transaction.channel_sequence, ChannelSequence(2));
        assert!(SessionModel::executes(exec.transaction.payload));
        assert!(!SessionModel::executes(delete.transaction.payload));
    }

    /// The refusal that keeps the rest of the model honest.
    #[test]
    fn a_command_with_no_established_contract_never_becomes_a_transaction() {
        let mut s = SessionModel::new(SessionId(1));
        // CmdDelay: judged, unresolved.
        let err = s.admit(&packet(0x3d)).expect_err("unresolved is refused");
        assert!(matches!(err, Refusal::UnestablishedContract { .. }));
        assert_eq!(err.slug(), "ingress_unestablished_contract");
        // An opcode no dispatch table declares refuses differently, because the
        // two are different problems and only one is closed by writing a
        // handler.
        let err = s.admit(&packet(0x1d)).expect_err("undeclared is refused");
        assert!(matches!(err, Refusal::UnknownCommand { .. }));
        assert_eq!(s.refusals(), 2);
    }

    /// A refusal must leave no gap in either order, or a reader of the ordinals
    /// has to explain a hole that means nothing.
    #[test]
    fn a_refusal_consumes_no_ordinal_and_no_sequence() {
        let mut s = SessionModel::new(SessionId(1));
        s.admit(&packet(0x37)).expect("accepted");
        s.admit(&packet(0x3d)).expect_err("refused");
        let next = s.admit(&packet(0x37)).expect("accepted");
        assert_eq!(next.transaction.ingress, IngressOrdinal(2));
        assert_eq!(next.transaction.channel_sequence, ChannelSequence(2));
    }

    /// Channel sequences are per domain; the ingress ordinal is not.
    #[test]
    fn two_domains_keep_separate_sequences_in_one_arrival_order() {
        let mut s = SessionModel::new(SessionId(1));
        let mut a = packet(0x37);
        a.domain = ChannelId(2);
        let mut b = packet(0x37);
        b.domain = ChannelId(3);
        let first = s.admit(&a).expect("accepted");
        let second = s.admit(&b).expect("accepted");
        let third = s.admit(&a).expect("accepted");
        assert_eq!(
            [
                first.transaction.ingress,
                second.transaction.ingress,
                third.transaction.ingress
            ],
            [IngressOrdinal(1), IngressOrdinal(2), IngressOrdinal(3)]
        );
        assert_eq!(second.transaction.channel_sequence, ChannelSequence(1));
        assert_eq!(third.transaction.channel_sequence, ChannelSequence(2));
    }

    #[test]
    fn hazards_and_completion_travel_together() {
        let mut s = SessionModel::new(SessionId(1));
        let mut writer = packet(0x37);
        writer.accesses = vec![whole(1, AccessMode::Write)];
        writer.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(1),
        });
        let mut reader = packet(0x37);
        reader.accesses = vec![whole(1, AccessMode::Read)];

        let w = s.admit(&writer).expect("accepted");
        assert!(w.ready);
        let r = s.admit(&reader).expect("accepted");
        assert!(!r.ready);
        assert_eq!(r.hazard_waits, vec![w.transaction.ingress]);
        assert_eq!(s.take_ready(), vec![w.transaction.ingress]);

        s.complete(w.transaction.ingress);
        assert_eq!(s.take_ready(), vec![r.transaction.ingress]);
        assert_eq!(
            s.scheduler().published_value(StampSlot(0)),
            Some(StampValue(1)),
            "completion published the stamp as well as releasing the hazard"
        );
        // And the completed transaction stops ordering later work.
        let mut later = packet(0x37);
        later.accesses = vec![whole(1, AccessMode::Write)];
        let l = s.admit(&later).expect("accepted");
        assert_eq!(l.hazard_waits, vec![r.transaction.ingress]);
    }

    /// A reset opens a new lifetime and does not throw away work that has not
    /// completed. Dropping it here would be a reset that loses a completion the
    /// host is still going to deliver.
    #[test]
    fn a_reset_opens_a_generation_without_abandoning_accepted_work() {
        let mut s = SessionModel::new(SessionId(1));
        let mut writer = packet(0x37);
        writer.accesses = vec![whole(1, AccessMode::Write)];
        let w = s.admit(&writer).expect("accepted");
        let before = s.generation();
        let after = s.reset();
        assert!(after > before);
        assert_eq!(s.scheduler().pending(), 1);
        // Work accepted after the reset carries the new generation and still
        // orders against the old transaction, which has not completed.
        let mut reader = packet(0x37);
        reader.accesses = vec![whole(1, AccessMode::Read)];
        let r = s.admit(&reader).expect("accepted");
        assert_eq!(r.transaction.session, after);
        assert_eq!(r.hazard_waits, vec![w.transaction.ingress]);
    }

    #[test]
    fn a_session_carries_the_identity_it_was_created_with() {
        let s = SessionModel::new(SessionId(4));
        assert_eq!(s.id(), SessionId(4));
        assert_eq!(s.generation(), SessionGeneration::FIRST);
    }

    /// Two of the retired slots and an acknowledged no-op are all `Control`,
    /// and all of them are still transactions: they retire stamps, and
    /// something has to order that.
    #[test]
    fn the_acknowledged_noops_are_transactions_like_any_other() {
        let mut s = SessionModel::new(SessionId(1));
        for opcode in [0x1e, 0x03, 0x32] {
            let t = s.admit(&packet(opcode)).expect("accepted");
            assert_eq!(t.transaction.payload, PayloadClass::Control);
            assert!(t.ready);
        }
    }
}
