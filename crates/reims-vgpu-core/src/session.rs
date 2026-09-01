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
//! It owns the ordinal counters, the per-channel sequences, the hazard graph,
//! the readiness service and each channel's publication order. It owns no
//! resources, no pipelines and no host objects, and it cannot: those live
//! behind a lease whose identity carries this session's generation, and the
//! crate they live in is not this one.

use crate::depend::DependencyGraph;
use crate::identity::{
    ChannelId, ChannelSequence, CompletionStamp, DeviceEpoch, IngressOrdinal, ResourceId,
    SessionGeneration, SessionId, StampWait, TransactionIdentity,
};
use crate::publish::{Publisher, Release, RetireRefusal};
use crate::ready::Scheduler;
use crate::retire::Lifetime;
use crate::transaction::{classify, DeviceTransaction, Payload, PayloadClass};
use reims_vgpu_protocol::packets::{find, Channel};
use std::collections::{BTreeMap, BTreeSet};

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
    /// The host device incarnation ended and no replacement exists yet.
    /// Admitting would promise ordering and completion on a device that is not
    /// there. Not a guest error and not a semantic one, which is why it is
    /// neither of the other two.
    DeviceLost { epoch: DeviceEpoch },
    /// A replacement device was asked for while the current one is live.
    DeviceNotLost { epoch: DeviceEpoch },
    /// The packet named a submission domain no channel definition opened.
    /// Admitting it would give it an ordering position in a publication order
    /// nothing will ever drain, which is a completion word the guest waits on
    /// forever.
    ChannelNotOpen { channel: ChannelId },
    /// A channel definition named a domain that is already open. Silently
    /// reopening would reset a publication order that still has positions in
    /// it.
    ChannelAlreadyOpen { channel: ChannelId },
    /// The packet arrived after the semantic lifetime it names was closed.
    /// Not an error in the guest: a reset races in-flight submissions, and the
    /// contract is that the closed generation stops accepting rather than that
    /// the guest stops sending.
    GenerationClosed {
        named: SessionGeneration,
        current: SessionGeneration,
    },
    /// The opcode declares one payload class and the decoded payload is
    /// another. Admitting it would order the packet as the wrong kind of work
    /// against a namespace that does not own what it names.
    PayloadMismatch {
        channel: Channel,
        opcode: u16,
        declared: PayloadClass,
        decoded: PayloadClass,
    },
}

impl Refusal {
    pub const fn slug(self) -> &'static str {
        match self {
            Self::UnknownCommand { .. } => "ingress_unknown_command",
            Self::UnestablishedContract { .. } => "ingress_unestablished_contract",
            Self::DeviceLost { .. } => "ingress_device_lost",
            Self::DeviceNotLost { .. } => "ingress_device_not_lost",
            Self::ChannelNotOpen { .. } => "ingress_channel_not_open",
            Self::ChannelAlreadyOpen { .. } => "ingress_channel_already_open",
            Self::PayloadMismatch { .. } => "ingress_payload_mismatch",
            Self::GenerationClosed { .. } => "ingress_generation_closed",
        }
    }
}

/// Whether this session has a host device incarnation to execute on.
///
/// Separate from the epoch identity: the identity says *which* incarnation,
/// this says whether there is one. A session between a loss and its
/// replacement has an epoch that names something dead, and admitting work
/// against it would be promising execution on a device that does not exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceState {
    Live,
    Lost,
}

/// A packet as ingress receives it, before it is a transaction.
///
/// It carries no position. That is the point: the ordinal and the channel
/// sequence are consumed under the arrival this call *is*, so a caller that
/// could state them could state ones that never happened. [`SessionModel::admit`]
/// assigns them and stamps the payload with them.
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
    /// The decoded work and everything it touches. Resolution is the caller's:
    /// it needs the namespaces, and this is the ordering plane.
    pub payload: Payload,
    /// Pipelines this packet needs whose leases came back pending.
    ///
    /// Only the pending ones: a lease that was already ready is not a wait, and
    /// passing it would hold the transaction for a compilation that has already
    /// finished. Taking the leases is the caller's, because the pipeline table
    /// lives beside this plane rather than in it.
    pub pipeline_waits: Vec<ResourceId>,
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
    epoch: DeviceEpoch,
    device: DeviceState,
    next_ingress: IngressOrdinal,
    /// The domains a channel definition has opened. Separate from
    /// `channel_sequence` because a channel that is open and has carried no
    /// packet is a real state, and one that has carried packets and been freed
    /// must stop being nameable even while its positions drain.
    open_channels: BTreeSet<ChannelId>,
    channel_sequence: BTreeMap<ChannelId, ChannelSequence>,
    graph: DependencyGraph,
    scheduler: Scheduler,
    publisher: Publisher,
    /// Which channel position each admitted transaction holds, so completion
    /// can find its publication domain without the caller carrying it back.
    position: BTreeMap<IngressOrdinal, (ChannelId, ChannelSequence)>,
    refusals: usize,
}

impl SessionModel {
    #[must_use]
    pub fn new(id: SessionId) -> Self {
        Self {
            id,
            generation: SessionGeneration::FIRST,
            epoch: DeviceEpoch::FIRST,
            device: DeviceState::Live,
            next_ingress: IngressOrdinal::default().next(),
            open_channels: BTreeSet::new(),
            channel_sequence: BTreeMap::new(),
            graph: DependencyGraph::new(),
            scheduler: Scheduler::new(),
            publisher: Publisher::new(),
            position: BTreeMap::new(),
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

    /// The host device incarnation this session's native objects belong to.
    #[must_use]
    pub const fn epoch(&self) -> DeviceEpoch {
        self.epoch
    }

    /// The pair every lease this session issues carries.
    #[must_use]
    pub const fn lifetime(&self) -> Lifetime {
        Lifetime::new(self.generation, self.epoch)
    }

    /// Close this semantic lifetime and open the next.
    ///
    /// New resolution stops; accepted work is not invalidated. The ordering
    /// plane is deliberately *not* cleared: a transaction accepted in the old
    /// generation still has to complete, still publishes its stamp, and still
    /// releases whatever waited on it. Dropping it here would be a reset that
    /// can lose a completion the host is still going to deliver.
    ///
    /// The device epoch does not move. A guest reset says nothing about the
    /// host device, which may be perfectly healthy — recreating it here would
    /// throw away work the host is still executing in order to answer a
    /// question the guest did not ask.
    pub fn reset(&mut self) -> SessionGeneration {
        self.generation = self.generation.next();
        self.generation
    }

    /// Whether this session has a host device at all.
    #[must_use]
    pub const fn device_state(&self) -> DeviceState {
        self.device
    }

    /// End the host device incarnation.
    ///
    /// Every lease from this epoch becomes unusable at once: no timeline will
    /// advance to release it, because the thing that would advance it is what
    /// was lost. The semantic generation does not move — the guest has not
    /// reset, still names what it named, and is owed a typed terminal fact
    /// rather than a silent new lifetime.
    ///
    /// This does **not** open a replacement. Losing a device and having one
    /// again are two events, and folding them into one call would mean work
    /// submitted in between is admitted into an incarnation that does not
    /// exist. Until [`SessionModel::recreate_device`] runs, admission refuses.
    ///
    /// Returns the epoch that died, which is the identity the retirement queue
    /// has to be told about.
    pub fn device_lost(&mut self) -> DeviceEpoch {
        self.device = DeviceState::Lost;
        self.epoch
    }

    /// Open the next host device incarnation after a loss.
    ///
    /// # Errors
    ///
    /// If the device was never lost. A replacement created while the current
    /// incarnation is live would orphan every lease against a device that is
    /// still perfectly able to execute them.
    pub fn recreate_device(&mut self) -> Result<DeviceEpoch, Refusal> {
        if self.device == DeviceState::Live {
            return Err(Refusal::DeviceNotLost { epoch: self.epoch });
        }
        self.epoch = self.epoch.next();
        self.device = DeviceState::Live;
        Ok(self.epoch)
    }

    /// Turn a packet into a transaction, or refuse it.
    ///
    /// # Errors
    ///
    /// Returns the one check that refused. Nothing is mutated on a refusal —
    /// no ordinal is consumed and no sequence advances — so a refused packet
    /// leaves no gap in either order for a reader to explain.
    pub fn admit(&mut self, packet: &Packet) -> Result<Admitted, Refusal> {
        if self.device == DeviceState::Lost {
            self.refusals += 1;
            return Err(Refusal::DeviceLost { epoch: self.epoch });
        }
        let Some(judged) = find(packet.channel, packet.opcode) else {
            self.refusals += 1;
            return Err(Refusal::UnknownCommand {
                channel: packet.channel,
                opcode: packet.opcode,
            });
        };
        let Some(class) = classify(packet.channel, packet.opcode) else {
            debug_assert!(judged.closure.blocks_cutover());
            self.refusals += 1;
            return Err(Refusal::UnestablishedContract {
                channel: packet.channel,
                opcode: packet.opcode,
            });
        };
        // The opcode says which class the packet is and the payload says which
        // one it *became*. A decoder that resolved a delete into a present
        // would be resolving it against a namespace that does not own it, and
        // the transaction would then be ordered as the wrong kind of work.
        if packet.payload.class() != class {
            self.refusals += 1;
            return Err(Refusal::PayloadMismatch {
                channel: packet.channel,
                opcode: packet.opcode,
                declared: class,
                decoded: packet.payload.class(),
            });
        }

        if !self.open_channels.contains(&packet.domain) {
            self.refusals += 1;
            return Err(Refusal::ChannelNotOpen {
                channel: packet.domain,
            });
        }

        let ingress = self.next_ingress;
        self.next_ingress = ingress.next();
        let sequence = self
            .channel_sequence
            .entry(packet.domain)
            .or_default()
            .next();
        self.channel_sequence.insert(packet.domain, sequence);

        self.publisher.admit(packet.domain, sequence);
        self.position.insert(ingress, (packet.domain, sequence));

        let hazard_waits = self.graph.admit(ingress, packet.payload.accesses());
        let ready = self.scheduler.admit(
            ingress,
            &hazard_waits,
            &packet.stamp_waits,
            &packet.pipeline_waits,
            packet.completion,
        );
        Ok(Admitted {
            transaction: DeviceTransaction {
                identity: TransactionIdentity {
                    session: self.generation,
                    domain: packet.domain,
                    domain_sequence: sequence,
                    ingress,
                },
                stamp_waits: packet.stamp_waits.clone(),
                completion: packet.completion,
                payload: packet.payload.clone(),
            },
            hazard_waits,
            ready,
        })
    }

    /// Complete a transaction: release its dependents, stop its accesses
    /// creating edges, and hand its channel's publication order whatever it
    /// now owes.
    ///
    /// The first two halves are one call because they are one fact. A
    /// completion that released dependents without retiring accesses would
    /// leave a finished transaction ordering later work forever.
    ///
    /// The third is deliberately *not* the same fact. A stamp becomes readable
    /// when its channel's publication order reaches it, which may be now or
    /// may be after an earlier position finishes, so what comes back is what
    /// the channel actually published — possibly this transaction's stamp,
    /// possibly a queue of them, possibly nothing. Whatever it published is
    /// also published to the readiness service, because a packet waiting on a
    /// completion word waits for the word the guest would read.
    pub fn complete(&mut self, ingress: IngressOrdinal) -> Vec<Release> {
        let owed = self.scheduler.complete(ingress);
        self.graph.retire(ingress);
        let (domain, sequence) = self
            .position
            .remove(&ingress)
            .expect("completing a transaction that holds no channel position");
        let released = self.publisher.complete(domain, sequence, owed);
        for release in &released {
            if let Some(stamp) = release.stamp {
                self.scheduler.publish(stamp);
            }
        }
        released
    }

    /// Remove a transaction that will never publish, releasing whatever its
    /// channel was holding behind it.
    ///
    /// A position that cannot finish still holds its channel's head, so
    /// something has to take it out. Nothing here decides *that* it cannot
    /// finish; the caller does, and says so on its failure channel.
    pub fn withdraw(&mut self, ingress: IngressOrdinal) -> Vec<Release> {
        let (domain, sequence) = self
            .position
            .remove(&ingress)
            .expect("withdrawing a transaction that holds no channel position");
        let released = self.publisher.withdraw(domain, sequence);
        for release in &released {
            if let Some(stamp) = release.stamp {
                self.scheduler.publish(stamp);
            }
        }
        released
    }

    #[must_use]
    pub const fn publisher(&self) -> &Publisher {
        &self.publisher
    }

    /// Open a submission domain, as a channel definition does.
    ///
    /// A domain has to be opened before anything may be admitted to it.
    /// Creating it on first use instead would mean a packet naming a channel
    /// the guest never defined gets an ordering position and a completion
    /// obligation in a publication order nothing drains — and the guest waits
    /// on that word forever. Which integer the root channel is, is not decided
    /// here: the caller that opened the ring knows it and opens it like any
    /// other.
    ///
    /// # Errors
    ///
    /// If the domain is already open.
    pub fn open_channel(&mut self, domain: ChannelId) -> Result<(), Refusal> {
        if !self.open_channels.insert(domain) {
            self.refusals += 1;
            return Err(Refusal::ChannelAlreadyOpen { channel: domain });
        }
        Ok(())
    }

    /// Whether a domain is open.
    #[must_use]
    pub fn channel_open(&self, domain: ChannelId) -> bool {
        self.open_channels.contains(&domain)
    }

    /// End a channel's publication lifetime, as a channel free does.
    ///
    /// # Errors
    ///
    /// If the channel still holds unreleased positions. A free that dropped
    /// them would drop the completion words the guest is waiting on, so the
    /// caller drains first.
    pub fn retire_channel(&mut self, domain: ChannelId) -> Result<(), RetireRefusal> {
        self.publisher.retire(domain)?;
        self.channel_sequence.remove(&domain);
        self.open_channels.remove(&domain);
        Ok(())
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
    use crate::access::{AccessIntent, AccessKey, AccessMode, BackingId, ResourceKey};
    use crate::identity::{StampSlot, StampValue};
    use crate::retire::Validity;

    /// A session with the two submission domains the tests use already open,
    /// because opening them is a channel definition's job and not a thing under
    /// test here. `channel_lifetime` tests the opening itself.
    fn session() -> SessionModel {
        let mut s = SessionModel::new(SessionId(1));
        s.open_channel(ChannelId(2)).expect("fresh");
        s.open_channel(ChannelId(3)).expect("fresh");
        s
    }

    /// A packet whose payload is the one its opcode's class calls for, with
    /// nothing in it. What is under test on this plane is ordering, not decode,
    /// so the payload is the emptiest lawful member of the right class — and it
    /// has to be the *right* class, because `admit` refuses a payload that
    /// disagrees with the opcode.
    fn packet(opcode: u16) -> Packet {
        Packet {
            channel: Channel::Child,
            domain: ChannelId(2),
            opcode,
            stamp_waits: Vec::new(),
            completion: None,
            payload: empty_payload(Channel::Child, opcode),
            pipeline_waits: Vec::new(),
        }
    }

    fn empty_payload(channel: Channel, opcode: u16) -> Payload {
        match classify(channel, opcode) {
            Some(PayloadClass::Exec) => Payload::Exec(crate::exec::ExecWork::default()),
            Some(PayloadClass::ResourceLifecycle) => Payload::ResourceLifecycle {
                op: crate::lifecycle::LifecycleOp::DeleteTask {
                    task: crate::identity::TaskId(1),
                },
                accesses: Vec::new(),
            },
            Some(PayloadClass::Query) => Payload::Query {
                request: crate::query::QueryRequest {
                    kind: crate::query::QueryKind::of(channel, opcode).expect("a query"),
                    destination: crate::query::ReplyDestination {
                        backing: BackingId(1),
                        bytes: crate::access::ByteRange {
                            offset: 0,
                            length: 64,
                        },
                    },
                },
                accesses: Vec::new(),
            },
            Some(PayloadClass::Present) => Payload::Present {
                form: crate::present::PresentForm::of(channel, opcode).expect("a present"),
                accesses: Vec::new(),
            },
            // A packet the model refuses never reaches its payload, so the
            // class it would have had is not a thing this can answer. `Nop` is
            // the emptiest payload there is, and the refusal happens first.
            Some(PayloadClass::Control) | None => {
                Payload::Control(crate::control::ControlOp::Inert {
                    kind: crate::control::ControlKind::of(channel, opcode)
                        .unwrap_or(crate::control::ControlKind::Nop),
                })
            }
        }
    }

    /// Give a packet's payload the accesses a test wants it to make.
    ///
    /// There is one list and the payload owns it, so this reaches into the
    /// payload rather than setting a field beside it.
    fn touching(mut packet: Packet, accesses: Vec<AccessIntent>) -> Packet {
        match &mut packet.payload {
            Payload::Exec(work) => work.accesses = accesses,
            Payload::ResourceLifecycle { accesses: a, .. }
            | Payload::Query { accesses: a, .. }
            | Payload::Present { accesses: a, .. } => *a = accesses,
            Payload::Control(_) => {
                assert!(accesses.is_empty(), "a control packet touches nothing");
            }
        }
        packet
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
        let mut s = session();
        let exec = s.admit(&packet(0x37)).expect("EXEC is accepted");
        let delete = s.admit(&packet(0x25)).expect("delete is accepted");
        assert_eq!(exec.transaction.class(), PayloadClass::Exec);
        assert_eq!(delete.transaction.class(), PayloadClass::ResourceLifecycle);
        assert_eq!(exec.transaction.identity.ingress, IngressOrdinal(1));
        assert_eq!(delete.transaction.identity.ingress, IngressOrdinal(2));
        assert_eq!(
            exec.transaction.identity.domain_sequence,
            ChannelSequence(1)
        );
        assert_eq!(
            delete.transaction.identity.domain_sequence,
            ChannelSequence(2)
        );
        assert!(SessionModel::executes(exec.transaction.class()));
        assert!(!SessionModel::executes(delete.transaction.class()));
    }

    /// The opcode declares a class and the payload arrives as one. If they can
    /// differ, they will: a decode that resolved a delete against the display's
    /// namespace would produce a `Present` under opcode `0x25`, and the
    /// transaction would then be ordered as a frame rather than a retirement.
    /// So `admit` compares them, and refuses rather than trusting either.
    #[test]
    fn a_payload_that_is_not_its_opcodes_class_is_refused() {
        let mut s = session();
        let mut wrong = packet(0x25);
        wrong.payload = Payload::Exec(crate::exec::ExecWork::default());
        let err = s.admit(&wrong).expect_err("a delete is not an EXEC");
        assert_eq!(
            err,
            Refusal::PayloadMismatch {
                channel: Channel::Child,
                opcode: 0x25,
                declared: PayloadClass::ResourceLifecycle,
                decoded: PayloadClass::Exec,
            }
        );
        assert_eq!(err.slug(), "ingress_payload_mismatch");
        // And it left no gap: the next packet takes position one.
        let next = s.admit(&packet(0x37)).expect("accepted");
        assert_eq!(next.transaction.identity.ingress, IngressOrdinal(1));
        assert_eq!(
            next.transaction.identity.domain_sequence,
            ChannelSequence(1)
        );
    }

    /// The envelope has no access list of its own, so the accesses the
    /// dependency graph ordered against are the ones the payload is holding —
    /// there is no second list for a caller to have filled differently.
    #[test]
    fn the_accesses_a_transaction_is_ordered_by_are_its_payloads() {
        let mut s = session();
        let w = s
            .admit(&touching(packet(0x37), vec![whole(1, AccessMode::Write)]))
            .expect("accepted");
        assert_eq!(w.transaction.accesses(), &[whole(1, AccessMode::Write)]);
        assert_eq!(
            w.transaction.payload.exec().expect("an EXEC").accesses,
            w.transaction.accesses(),
            "the envelope's answer is the payload's, not a copy of it"
        );
        // A reader of the same backing is ordered behind it, which is only
        // possible if the graph saw the payload's list.
        let r = s
            .admit(&touching(packet(0x37), vec![whole(1, AccessMode::Read)]))
            .expect("accepted");
        assert_eq!(r.hazard_waits, vec![w.transaction.identity.ingress]);
    }

    /// The executor's view of an admitted EXEC is derived from the envelope,
    /// so its identity is the envelope's by construction rather than by
    /// agreement. Before the split these were two stampings of one fact and a
    /// resolver assigned one of them.
    #[test]
    fn the_executors_view_of_an_exec_carries_the_identity_admission_assigned() {
        let mut s = session();
        s.admit(&packet(0x37)).expect("accepted");
        let second = s.admit(&packet(0x37)).expect("accepted").transaction;
        let view = second.exec().expect("an EXEC");
        assert_eq!(view.identity, second.identity);
        assert_eq!(view.ingress(), IngressOrdinal(2));
        assert_eq!(view.domain_sequence(), ChannelSequence(2));
        assert_eq!(view.domain(), ChannelId(2));
        assert!(
            core::ptr::eq(view.work, second.payload.exec().expect("an EXEC")),
            "the view borrows the envelope's work rather than copying it"
        );
        // And nothing else offers one.
        assert!(s
            .admit(&packet(0x25))
            .expect("accepted")
            .transaction
            .exec()
            .is_none());
    }

    /// [`Payload::Control`] has no access list, and that is a contract claim:
    /// opening a channel, moving a cursor and doing nothing touch no guest
    /// resource. A control packet with an access would be a decode error
    /// upstream, and it is not representable here.
    #[test]
    fn control_transactions_touch_no_resource() {
        let mut s = session();
        let mut seen = 0;
        for p in reims_vgpu_protocol::packets::LEDGER {
            if classify(p.channel, p.opcode) != Some(PayloadClass::Control) {
                continue;
            }
            let payload = empty_payload(p.channel, p.opcode);
            assert_eq!(payload.class(), PayloadClass::Control);
            assert!(
                payload.accesses().is_empty(),
                "{} {:#04x} ({}) is control and names a resource",
                p.channel.name(),
                p.opcode,
                p.name
            );
            seen += 1;
        }
        assert_eq!(seen, 23, "the twenty-three control packets");
        // And the graph agrees: a control packet creates no hazard edge against
        // a writer of anything.
        let w = s
            .admit(&touching(packet(0x37), vec![whole(1, AccessMode::Write)]))
            .expect("accepted");
        let nop = s.admit(&packet(0x1e)).expect("CmdNOP is accepted");
        assert!(
            nop.hazard_waits.is_empty(),
            "a control packet waits for nothing it does not touch"
        );
        assert!(w.transaction.accesses().len() == 1);
    }

    /// The refusal that keeps the rest of the model honest.
    #[test]
    fn a_command_with_no_established_contract_never_becomes_a_transaction() {
        let mut s = session();
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
        let mut s = session();
        s.admit(&packet(0x37)).expect("accepted");
        s.admit(&packet(0x3d)).expect_err("refused");
        let next = s.admit(&packet(0x37)).expect("accepted");
        assert_eq!(next.transaction.identity.ingress, IngressOrdinal(2));
        assert_eq!(
            next.transaction.identity.domain_sequence,
            ChannelSequence(2)
        );
    }

    /// Channel sequences are per domain; the ingress ordinal is not.
    #[test]
    fn two_domains_keep_separate_sequences_in_one_arrival_order() {
        let mut s = session();
        let mut a = packet(0x37);
        a.domain = ChannelId(2);
        let mut b = packet(0x37);
        b.domain = ChannelId(3);
        let first = s.admit(&a).expect("accepted");
        let second = s.admit(&b).expect("accepted");
        let third = s.admit(&a).expect("accepted");
        assert_eq!(
            [
                first.transaction.identity.ingress,
                second.transaction.identity.ingress,
                third.transaction.identity.ingress
            ],
            [IngressOrdinal(1), IngressOrdinal(2), IngressOrdinal(3)]
        );
        assert_eq!(
            second.transaction.identity.domain_sequence,
            ChannelSequence(1)
        );
        assert_eq!(
            third.transaction.identity.domain_sequence,
            ChannelSequence(2)
        );
    }

    #[test]
    fn hazards_and_completion_travel_together() {
        let mut s = session();
        let mut writer = touching(packet(0x37), vec![whole(1, AccessMode::Write)]);
        writer.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(1),
        });
        let reader = touching(packet(0x37), vec![whole(1, AccessMode::Read)]);

        let w = s.admit(&writer).expect("accepted");
        assert!(w.ready);
        let r = s.admit(&reader).expect("accepted");
        assert!(!r.ready);
        assert_eq!(r.hazard_waits, vec![w.transaction.identity.ingress]);
        assert_eq!(s.take_ready(), vec![w.transaction.identity.ingress]);

        let released = s.complete(w.transaction.identity.ingress);
        assert_eq!(s.take_ready(), vec![r.transaction.identity.ingress]);
        assert_eq!(
            released,
            vec![Release {
                sequence: ChannelSequence(1),
                stamp: writer.completion,
            }],
            "it was its channel's head, so its stamp published at once"
        );
        assert_eq!(
            s.scheduler().published_value(StampSlot(0)),
            Some(StampValue(1)),
            "and a packet waiting on that word may now run"
        );
        // And the completed transaction stops ordering later work.
        let later = touching(packet(0x37), vec![whole(1, AccessMode::Write)]);
        let l = s.admit(&later).expect("accepted");
        assert_eq!(l.hazard_waits, vec![r.transaction.identity.ingress]);
    }

    /// Out-of-order completion is ordinary; out-of-order publication is not.
    #[test]
    fn a_channel_publishes_in_its_own_order_however_the_work_finishes() {
        let mut s = session();
        let mut first = packet(0x37);
        first.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(1),
        });
        let mut second = packet(0x37);
        second.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(2),
        });
        let a = s.admit(&first).expect("accepted");
        let b = s.admit(&second).expect("accepted");

        assert!(
            s.complete(b.transaction.identity.ingress).is_empty(),
            "the second position finished first and published nothing"
        );
        assert_eq!(
            s.scheduler().published_value(StampSlot(0)),
            None,
            "a guest polling that word must not see the later value first"
        );
        assert_eq!(
            s.publisher().blocked(),
            vec![(ChannelId(2), 1)],
            "and the cost of holding it is counted"
        );
        assert_eq!(
            s.complete(a.transaction.identity.ingress)
                .into_iter()
                .map(|r| r.stamp)
                .collect::<Vec<_>>(),
            vec![first.completion, second.completion],
            "the head finishing publishes both, in channel order"
        );
    }

    /// A position that cannot finish must leave, or its channel never publishes
    /// again.
    #[test]
    fn withdrawing_a_head_releases_what_was_queued_behind_it() {
        let mut s = session();
        let mut second = packet(0x37);
        second.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(2),
        });
        let a = s.admit(&packet(0x37)).expect("accepted");
        let b = s.admit(&second).expect("accepted");
        assert!(s.complete(b.transaction.identity.ingress).is_empty());
        assert_eq!(
            s.withdraw(a.transaction.identity.ingress)
                .into_iter()
                .map(|r| r.stamp)
                .collect::<Vec<_>>(),
            vec![second.completion]
        );
        assert_eq!(
            s.scheduler().published_value(StampSlot(0)),
            Some(StampValue(2))
        );
    }

    #[test]
    fn a_channel_with_unpublished_work_cannot_end_its_lifetime() {
        let mut s = session();
        let a = s.admit(&packet(0x37)).expect("accepted");
        assert_eq!(
            s.retire_channel(ChannelId(2)),
            Err(RetireRefusal::LivePositions { outstanding: 1 })
        );
        s.complete(a.transaction.identity.ingress);
        assert_eq!(s.retire_channel(ChannelId(2)), Ok(()));
        assert!(
            !s.channel_open(ChannelId(2)),
            "a freed channel stops being nameable"
        );
        assert_eq!(
            s.admit(&packet(0x37)),
            Err(Refusal::ChannelNotOpen {
                channel: ChannelId(2)
            }),
            "and a packet naming it is refused rather than reopening it"
        );
        // A later definition of the channel starts at position one rather than
        // continuing the lifetime that just ended.
        s.open_channel(ChannelId(2)).expect("free again");
        let next = s.admit(&packet(0x37)).expect("accepted");
        assert_eq!(
            next.transaction.identity.domain_sequence,
            ChannelSequence(1)
        );
    }

    /// A packet naming a domain no definition opened is refused at ingress.
    /// Creating the domain on first use instead would give the packet an
    /// ordering position and a completion obligation in a publication order
    /// nothing drains, and the guest waits on that word forever.
    #[test]
    fn a_packet_on_an_undefined_channel_is_refused_and_consumes_nothing() {
        let mut s = SessionModel::new(SessionId(1));
        let before = s.refusals();
        assert_eq!(
            s.admit(&packet(0x37)),
            Err(Refusal::ChannelNotOpen {
                channel: ChannelId(2)
            })
        );
        assert_eq!(s.refusals(), before + 1);
        s.open_channel(ChannelId(2)).expect("fresh");
        let first = s.admit(&packet(0x37)).expect("accepted");
        assert_eq!(
            first.transaction.identity.ingress,
            IngressOrdinal::default().next(),
            "the refused packet consumed no ordinal"
        );
        assert_eq!(
            first.transaction.identity.domain_sequence,
            ChannelSequence(1)
        );
    }

    /// Reopening a live channel would reset a publication order that still has
    /// positions in it.
    #[test]
    fn a_channel_is_defined_once() {
        let mut s = session();
        assert_eq!(
            s.open_channel(ChannelId(2)),
            Err(Refusal::ChannelAlreadyOpen {
                channel: ChannelId(2)
            })
        );
    }

    /// The two transitions are independent, which is the whole reason they are
    /// two identities.
    #[test]
    fn a_guest_reset_and_a_device_loss_move_different_lifetimes() {
        let mut s = session();
        let start = s.lifetime();

        s.reset();
        assert_eq!(
            s.epoch(),
            start.epoch,
            "a guest reset says nothing about the host device, which may be \
             perfectly healthy"
        );
        assert_ne!(s.generation(), start.session);
        assert_eq!(s.device_state(), DeviceState::Live);

        let after_reset = s.lifetime();
        let died = s.device_lost();
        assert_eq!(died, after_reset.epoch, "the epoch that died is named");
        assert_eq!(
            s.generation(),
            after_reset.session,
            "the guest has not reset; it still names what it named"
        );
        assert_eq!(s.device_state(), DeviceState::Lost);

        let replacement = s.recreate_device().expect("lost, so replaceable");
        assert_ne!(replacement, died);
        assert_eq!(s.generation(), after_reset.session);
    }

    /// Work submitted between a loss and its replacement has nothing to run on
    /// and must be told so, not admitted into an incarnation that is gone.
    #[test]
    fn a_lost_device_refuses_admission_until_it_is_replaced() {
        let mut s = session();
        let epoch = s.device_lost();
        assert_eq!(
            s.admit(&packet(0x37)),
            Err(Refusal::DeviceLost { epoch }),
            "ordering and completion cannot be promised on a device that is \
             not there"
        );
        assert_eq!(s.refusals(), 1);
        s.recreate_device().expect("lost");
        assert!(s.admit(&packet(0x37)).is_ok());
    }

    #[test]
    fn a_live_device_cannot_be_replaced() {
        let mut s = session();
        assert_eq!(
            s.recreate_device(),
            Err(Refusal::DeviceNotLost { epoch: s.epoch() }),
            "a replacement made while the device is live would orphan every \
             lease against a device still able to execute them"
        );
    }

    /// A lease issued before either transition is judged on both, separately.
    #[test]
    fn a_lease_from_before_both_transitions_is_judged_on_both() {
        let mut s = session();
        let lease = s.lifetime();
        assert_eq!(lease.against(s.generation(), s.epoch()), Validity::Live);

        s.reset();
        let after = lease.against(s.generation(), s.epoch());
        assert!(!after.admits_new_work(), "the guest may not name it");
        assert!(
            after.handles_usable(),
            "and the submission the host is still executing must finish"
        );

        s.device_lost();
        s.recreate_device().expect("lost");
        assert_eq!(lease.against(s.generation(), s.epoch()), Validity::Gone);
    }

    /// Two attached devices are two values with nothing between them. The test
    /// exists so that adding a process-global anywhere in this plane fails
    /// here rather than in a guest.
    #[test]
    fn one_sessions_reset_or_loss_cannot_reach_another_session() {
        let mut a = session();
        let mut b = SessionModel::new(SessionId(2));
        b.open_channel(ChannelId(2)).expect("fresh");
        let untouched = b.lifetime();
        let admitted = b.admit(&packet(0x37)).expect("accepted");

        a.reset();
        a.device_lost();
        a.recreate_device().expect("lost");

        assert_eq!(b.lifetime(), untouched);
        assert_eq!(b.device_state(), DeviceState::Live);
        assert_eq!(untouched.against(b.generation(), b.epoch()), Validity::Live);
        assert!(b.admit(&packet(0x37)).is_ok());
        assert!(!b.complete(admitted.transaction.identity.ingress).is_empty());
    }

    /// A reset opens a new lifetime and does not throw away work that has not
    /// completed. Dropping it here would be a reset that loses a completion the
    /// host is still going to deliver.
    #[test]
    fn a_reset_opens_a_generation_without_abandoning_accepted_work() {
        let mut s = session();
        let writer = touching(packet(0x37), vec![whole(1, AccessMode::Write)]);
        let w = s.admit(&writer).expect("accepted");
        let before = s.generation();
        let after = s.reset();
        assert!(after > before);
        assert_eq!(s.scheduler().pending(), 1);
        // Work accepted after the reset carries the new generation and still
        // orders against the old transaction, which has not completed.
        let reader = touching(packet(0x37), vec![whole(1, AccessMode::Read)]);
        let r = s.admit(&reader).expect("accepted");
        assert_eq!(r.transaction.identity.session, after);
        assert_eq!(r.hazard_waits, vec![w.transaction.identity.ingress]);
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
        let mut s = session();
        for opcode in [0x1e, 0x03, 0x32] {
            let t = s.admit(&packet(opcode)).expect("accepted");
            assert_eq!(t.transaction.class(), PayloadClass::Control);
            assert!(t.ready);
        }
    }
}
