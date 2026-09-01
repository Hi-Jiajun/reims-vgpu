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

use crate::control::{ChannelTransition, ControlOp};
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
    /// The packet binds a pipeline it can never use: one this device refused
    /// to build, or one this generation does not have.
    ///
    /// Refused rather than admitted with a wait, because the wait would never
    /// resolve. Admitting it and withdrawing it later costs the guest the same
    /// frame and costs this device a position it has to remember to take back.
    PipelineUnusable(crate::pipeline::LeaseRefusal),
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
            Self::PipelineUnusable(refusal) => refusal.slug(),
            Self::GenerationClosed { .. } => "ingress_generation_closed",
        }
    }
}

/// Why a control operation's transition did not happen.
///
/// Two variants because there are two owners, and neither refusal is invented
/// here: opening is this model's own [`Refusal::ChannelAlreadyOpen`] and
/// freeing is the publisher's [`RetireRefusal::LivePositions`]. Folding them
/// into one reason would lose which half of a channel's lifetime went wrong,
/// and restating either would be a second copy of a check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlRefusal {
    Open(Refusal),
    Free(RetireRefusal),
}

impl ControlRefusal {
    /// The name this reaches a failure channel under: the forwarded owner's,
    /// unchanged.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Open(refusal) => refusal.slug(),
            Self::Free(refusal) => refusal.slug(),
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
    /// The semantic lifetime this packet was **read under**.
    ///
    /// Not the one it will be admitted into: a reset races the drain, so a
    /// packet that left the ring before the reset and reaches ingress after it
    /// names a lifetime that has closed. Nothing else can tell — the guest's
    /// packet carries no generation, and by the time this model sees it, its own
    /// generation has already moved — so the reader states the one it was
    /// holding when it took the bytes, which is the one fact the reader has and
    /// this plane does not.
    ///
    /// See [`Refusal::GenerationClosed`], and
    /// [`crate::interpret::Refusal::StaleGeneration`], which is the serial
    /// reference's spelling of the same rule.
    pub session: SessionGeneration,
    pub opcode: u16,
    pub stamp_waits: Vec<StampWait>,
    pub completion: Option<CompletionStamp>,
    /// The decoded work and everything it touches. Resolution is the caller's:
    /// it needs the namespaces, and this is the ordering plane.
    pub payload: Payload,
}

/// What a device loss ended.
///
/// The epoch and the work are one value because they are one event: an epoch
/// that died without its stranded transactions being taken out is a set of
/// channels that never publish again, and a list of stranded transactions
/// without the epoch is a retirement queue that does not know what to abandon.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "the guest is owed a typed reason for every packet the loss stranded"]
pub struct DeviceLoss {
    /// The incarnation that ended.
    pub epoch: DeviceEpoch,
    /// Transactions admitted into it that can never complete, in ingress order.
    /// Already withdrawn from every plane; what is left is to name each one.
    pub stranded: Vec<IngressOrdinal>,
    /// Completion words the withdrawals released — work that had *already*
    /// completed and was waiting behind a stranded position for its channel's
    /// head. Those are not lost: the host delivered them before the device
    /// died and the guest is owed them.
    pub released: Vec<Release>,
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
    /// The pipeline objects this session's work binds.
    ///
    /// Held here rather than beside this plane, because the waits a transaction
    /// is admitted with are the table's answer about that transaction's own
    /// leases. A table on the other side of the boundary means a caller states
    /// the waits, and a caller that states them can state ones the payload does
    /// not lease or omit ones it does.
    pipelines: crate::pipeline::PipelineTable,
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
            pipelines: crate::pipeline::PipelineTable::new(),
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
    /// Returns the epoch that died — the identity the retirement queue has to
    /// be told about — and the work that died with it.
    ///
    /// **Every transaction admitted into that epoch is stranded.** Nothing will
    /// complete them, because the thing that would is what was lost, and each
    /// one holds a position in the publication order, the dependency graph and
    /// the readiness service. Left there, the channel never publishes again and
    /// every later transaction sharing a backing with one of them waits forever
    /// — so they are withdrawn here rather than named and left for a caller to
    /// remember, which is also the only thing that *can* withdraw them: the
    /// positions are this model's and a caller cannot enumerate them.
    ///
    /// They still come back. The guest is owed a typed terminal fact for each
    /// packet it will never see completed, and this model has no failure
    /// channel to put one on.
    pub fn device_lost(&mut self) -> DeviceLoss {
        self.device = DeviceState::Lost;
        // Ingress order, so a report reads in the order the guest sent them.
        let stranded: Vec<IngressOrdinal> = self.position.keys().copied().collect();
        let mut released = Vec::new();
        for ingress in &stranded {
            released.extend(self.withdraw(*ingress));
        }
        DeviceLoss {
            epoch: self.epoch,
            stranded,
            released,
        }
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
        // Before the packet's own contract is judged, because this is not about
        // the packet: it is a lifetime question, and the objects the packet
        // names no longer exist whatever its opcode says.
        if packet.session != self.generation {
            self.refusals += 1;
            return Err(Refusal::GenerationClosed {
                named: packet.session,
                current: self.generation,
            });
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

        // Shape before content, and before anything is charged: a domain no
        // channel definition opened is an envelope fact like the generation and
        // the payload class above it. It used to be checked *after* the
        // pipeline leases below, which took — and charged the census for —
        // leases for a packet that was then refused, so the number that says
        // whether compilation starts early enough grew with refused packets.
        if !self.open_channels.contains(&packet.domain) {
            self.refusals += 1;
            return Err(Refusal::ChannelNotOpen {
                channel: packet.domain,
            });
        }

        // The waits are the table's answer about this payload's own leases, not
        // a list the caller brought. A caller that could state them could state
        // ones the payload does not lease — parking the transaction for a
        // compilation it has no interest in, which the guest experiences as a
        // frame that never arrives — or omit ones it does, which runs a draw
        // against a pipeline that is still being built.
        //
        // Non-EXEC payloads lease nothing, so they wait for nothing: only GPU
        // work binds a pipeline.
        let generation = self.generation;
        let pipeline_waits = match packet.payload.exec() {
            Some(work) => self.pipelines.waits_for(&work.pipeline_leases, generation),
            None => Ok(Vec::new()),
        };
        let pipeline_waits = match pipeline_waits {
            Ok(waits) => waits,
            Err(refusal) => {
                self.refusals += 1;
                return Err(Refusal::PipelineUnusable(refusal));
            }
        };

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
            &pipeline_waits,
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

    /// Remove a transaction that will never publish, releasing everything it
    /// was holding.
    ///
    /// A transaction that cannot finish holds a position in **three** planes,
    /// and taking it out of one is not taking it out. Its channel's publication
    /// head is the visible one. The other two are the ones that hang: its
    /// accesses stay live in the dependency graph, so every later transaction
    /// touching that memory takes a hazard wait on an ordinal nothing will ever
    /// complete; and it stays pending in the readiness service, which is the
    /// only thing that decrements a dependent's remaining hazards.
    ///
    /// This used to release the first and neither of the other two, so
    /// withdrawing a transaction to un-stall a channel stalled every later one
    /// that shared a backing with it — the exact failure the withdrawal exists
    /// to prevent, moved from one plane to another.
    ///
    /// Nothing here decides *that* it cannot finish; the caller does, and says
    /// so on its failure channel.
    ///
    /// **Its own completion word is not published.** The work never ran, and a
    /// stamp published for it is a value the guest acts on. What the guest is
    /// owed instead is the typed reason, which is why the caller names one.
    pub fn withdraw(&mut self, ingress: IngressOrdinal) -> Vec<Release> {
        // Releases this transaction's dependents and forgets what it was
        // waiting on. The stamp it owed comes back and is deliberately dropped:
        // publication is `complete`'s and this is not a completion.
        let _never_published = self.scheduler.complete(ingress);
        self.graph.retire(ingress);
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

    /// Open a submission domain that no guest command asked for.
    ///
    /// A domain has to be opened before anything may be admitted to it.
    /// Creating it on first use instead would mean a packet naming a channel
    /// the guest never defined gets an ordering position and a completion
    /// obligation in a publication order nothing drains — and the guest waits
    /// on that word forever.
    ///
    /// **This is the bootstrap door, not the guest's.** The root ring exists
    /// before the guest can name anything, so whoever opened it opens its
    /// domain like any other; every domain the guest itself defines arrives as
    /// a `CmdDefineFifo` and goes through [`Self::apply_control`].
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

    /// End a channel's publication lifetime.
    ///
    /// The bootstrap door's other half — see [`Self::open_channel`]. A guest's
    /// `CmdFreeFifo` reaches it through [`Self::apply_control`].
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

    /// Perform a control operation's effect on this session.
    ///
    /// **The join between a resolved control packet and the state it changes.**
    /// [`crate::control::resolve`] turned the guest's bytes into a
    /// [`ControlOp`] and this model held the channel set, and nothing carried
    /// one to the other — so a guest's `CmdDefineFifo` decoded into an
    /// operation whose entire effect, opening the domain its next packet names,
    /// nobody performed. Every packet on that channel is then refused
    /// [`Refusal::ChannelNotOpen`], which is a device that answers a correct
    /// guest with a wall.
    ///
    /// The two other operation shapes are `Ok(())` and that is the claim, not
    /// an omission: [`ControlOp::Inert`]'s payload does nothing and
    /// [`ControlOp::Display`]'s belongs to the layer that has a display.
    /// Neither touches ordering, which is what this model owns. Matching
    /// exhaustively is what makes a fourth shape a compile error here rather
    /// than a silently ignored command.
    ///
    /// The envelope is *not* this call's business. A control transaction takes
    /// its ordering position and publishes its completion word like every other
    /// class, whether its transition happened or not — which is why a refusal
    /// here is a value the caller reports and not a reason to withhold a stamp.
    ///
    /// # Errors
    ///
    /// [`ControlRefusal`], forwarding whichever owner refused: a definition
    /// naming a domain that is already open, or a free naming one that still
    /// owes publication.
    pub fn apply_control(&mut self, op: ControlOp) -> Result<(), ControlRefusal> {
        match op {
            ControlOp::Channel {
                transition: ChannelTransition::Open,
                domain,
            } => self.open_channel(domain).map_err(ControlRefusal::Open),
            ControlOp::Channel {
                transition: ChannelTransition::Free,
                domain,
            } => self.retire_channel(domain).map_err(ControlRefusal::Free),
            ControlOp::Display { .. } | ControlOp::Inert { .. } => Ok(()),
        }
    }

    /// The pipeline objects this session's work binds, for the layer that
    /// declares them and advances their compilation.
    ///
    /// Read and write, because declaring a pipeline and stepping it through
    /// translation are the compiling layer's and not this plane's. What this
    /// plane owns is the consequence, which is why the two steps that *have* a
    /// consequence — a pipeline becoming usable and a pipeline becoming
    /// impossible — are [`Self::pipeline_ready`] and [`Self::pipeline_refused`]
    /// rather than calls on the table.
    pub const fn pipelines(&mut self) -> &mut crate::pipeline::PipelineTable {
        &mut self.pipelines
    }

    /// A pipeline finished building: record it, and release what was held for
    /// it.
    ///
    /// **The two halves are one call because they are one fact.** The table
    /// knew a pipeline had become `Ready` and the scheduler knew which
    /// transactions were parked on it, and nothing carried one to the other —
    /// so a transaction admitted with a pipeline wait was admitted into a wait
    /// nothing could ever discharge. It holds its channel's publication head,
    /// and every completion word behind it stops arriving.
    ///
    /// Returns whether the step was legal and taken. An illegal one is real: a
    /// compile that finishes after the guest deleted the pipeline, which must
    /// not resurrect it — and must not release work either, since that work
    /// cannot be admitted against a retired pipeline in the first place.
    pub fn pipeline_ready(&mut self, pipeline: ResourceId) -> bool {
        if !self
            .pipelines
            .advance(pipeline, crate::pipeline::PipelineState::Ready)
        {
            return false;
        }
        self.scheduler.pipeline_ready(pipeline);
        true
    }

    /// A pipeline will never build: record the reason, and name the
    /// transactions that can therefore never be ready.
    ///
    /// They come back rather than being made ready or dropped. Made ready they
    /// would execute against a pipeline that does not exist; dropped they would
    /// hold their channel's publication head forever. The caller withdraws each
    /// one — see [`Self::withdraw`] — and says why on its failure channel.
    ///
    /// Empty when the refusal was not a legal step, for
    /// [`Self::pipeline_ready`]'s reason.
    pub fn pipeline_refused(
        &mut self,
        pipeline: ResourceId,
        reason: crate::pipeline::RefusalReason,
    ) -> Vec<IngressOrdinal> {
        if !self.pipelines.refuse(pipeline, reason) {
            return Vec::new();
        }
        self.scheduler.pipeline_refused(pipeline)
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
    use crate::identity::{ObjectListRef, SlotGeneration, StampSlot, StampValue};
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
            // The tests here drive one lifetime; `reset` has its own.
            session: SessionGeneration::FIRST,
            opcode,
            stamp_waits: Vec::new(),
            completion: None,
            payload: empty_payload(Channel::Child, opcode),
        }
    }

    /// **A refused admission takes nothing.**
    ///
    /// `admit` promises that nothing is mutated on a refusal, and the pipeline
    /// leases were the exception: they were taken before the channel-open check
    /// below them, so a packet naming a domain no channel definition opened
    /// charged the census for leases the transaction never held. That number is
    /// what says whether starting compilation at declaration is early enough,
    /// and one that grows with refused packets cannot answer it.
    #[test]
    fn a_refused_admission_takes_no_pipeline_lease() {
        let mut s = SessionModel::new(SessionId(1));
        s.open_channel(ChannelId(2)).expect("fresh");
        let pipe = ResourceId {
            slot: ObjectListRef(9),
            generation: SlotGeneration(1),
        };
        s.pipelines().declare(pipe, SessionGeneration::FIRST);
        let before = s.pipelines().census();

        // A domain no definition opened.
        let mut closed = packet(0x37);
        closed.domain = ChannelId(7);
        if let Payload::Exec(work) = &mut closed.payload {
            work.pipeline_leases.push(pipe);
        }
        assert_eq!(
            s.admit(&closed),
            Err(Refusal::ChannelNotOpen {
                channel: ChannelId(7)
            })
        );
        assert_eq!(s.pipelines().census(), before, "a refusal took a lease");

        // And a lease list that refuses part way charges nothing for the part
        // ahead of the refusal.
        let absent = ResourceId {
            slot: ObjectListRef(10),
            generation: SlotGeneration(1),
        };
        let mut partial = packet(0x37);
        if let Payload::Exec(work) = &mut partial.payload {
            work.pipeline_leases.push(pipe);
            work.pipeline_leases.push(absent);
        }
        assert!(matches!(
            s.admit(&partial),
            Err(Refusal::PipelineUnusable(_))
        ));
        assert_eq!(
            s.pipelines().census(),
            before,
            "the pipelines ahead of the refused one were charged"
        );

        // The lease the admitted packet does hold is counted.
        let mut good = packet(0x37);
        if let Payload::Exec(work) = &mut good.payload {
            work.pipeline_leases.push(pipe);
        }
        s.admit(&good).expect("open domain, declared pipeline");
        assert_eq!(
            s.pipelines().census().leases_pending,
            before.leases_pending + 1
        );
    }

    /// An access naming no memory: the vocabulary for a target that could not
    /// be resolved.
    fn domain_only(domain: ChannelId) -> AccessIntent {
        AccessIntent {
            domain,
            key: crate::access::AccessKey::DomainOnly,
            mode: crate::access::AccessMode::Read,
            api_stages: 0,
            input_content_version: None,
            output_content_version: None,
        }
    }

    fn empty_payload(channel: Channel, opcode: u16) -> Payload {
        match classify(channel, opcode) {
            Some(PayloadClass::Exec) => Payload::Exec(crate::exec::ExecWork::default()),
            Some(PayloadClass::ResourceLifecycle) => Payload::ResourceLifecycle(
                crate::transaction::LifecyclePayload::new(
                    crate::lifecycle::LifecycleOp::DeleteTask {
                        task: crate::identity::TaskId(1),
                    },
                    Vec::new(),
                )
                .expect("a task teardown names no resource"),
            ),
            Some(PayloadClass::Query) => Payload::Query(crate::transaction::QueryPayload::new(
                crate::query::QueryRequest {
                    kind: crate::query::QueryKind::of(channel, opcode).expect("a query"),
                    destination: crate::query::ReplyDestination {
                        backing: BackingId(1),
                        bytes: crate::access::ByteRange {
                            offset: 0,
                            length: 64,
                        },
                    },
                    reply: crate::query::ReplyShape::Fixed { bytes: 16 },
                },
                ChannelId(0),
                None,
            )),
            Some(PayloadClass::Present) => {
                let packet = crate::present::resolve(channel, opcode, &0u32.to_le_bytes())
                    .expect("a present with a trailer");
                // The target is unresolved in this fixture, and a present that
                // named nothing at all would be refused — see `PresentPayload`.
                Payload::Present(
                    crate::transaction::PresentPayload::new(
                        packet,
                        vec![(packet.mapping, domain_only(ChannelId(0)))],
                    )
                    .expect("one read of the packet's own target"),
                )
            }
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
            // Its accesses must all be reads of the packet's own target, so
            // they are rebuilt through the payload rather than assigned.
            Payload::Present(present) => {
                let packet = *present.packet();
                *present = crate::transaction::PresentPayload::new(
                    packet,
                    accesses.into_iter().map(|a| (packet.mapping, a)).collect(),
                )
                .expect("reads of the packet's own target");
            }
            // The teardown `empty_payload` builds names no resource, so its
            // access list is unconstrained — but it is still the payload's, and
            // it is rebuilt rather than reached into.
            Payload::ResourceLifecycle(lifecycle) => {
                *lifecycle = crate::transaction::LifecyclePayload::new(
                    lifecycle.op().clone(),
                    accesses
                        .into_iter()
                        .map(|a| {
                            (
                                ResourceId {
                                    slot: ObjectListRef(0),
                                    generation: SlotGeneration(1),
                                },
                                a,
                            )
                        })
                        .collect(),
                )
                .expect("the fixture's operation names no resource");
            }
            // A query's access is its reply window and is not the test's to
            // choose — see `QueryPayload`.
            Payload::Query(query) => assert_eq!(
                accesses,
                vec![*query.access()],
                "a query touches its destination and nothing else"
            ),
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

    /// A transaction's pipeline waits are the table's answer about its own
    /// leases, and nothing else can state them.
    ///
    /// The two lists used to be built in different places — the leases as the
    /// records resolved, the waits from whatever the caller asked the table —
    /// and nothing tied them together, so a wait for a pipeline the packet
    /// never binds was representable, as was a packet that bound one and waited
    /// for nothing. The first parks the transaction for a compilation it has no
    /// interest in; the second runs a draw against a pipeline still being
    /// built.
    #[test]
    fn a_transactions_pipeline_waits_are_the_pipelines_it_binds() {
        let mut s = session();
        let pipeline = ResourceId {
            slot: ObjectListRef(9),
            generation: SlotGeneration(1),
        };
        let gen = s.generation();
        s.pipelines().declare(pipeline, gen);

        // Binding nothing waits for nothing, whatever the table holds.
        assert!(s.admit(&packet(0x37)).expect("accepted").ready);

        let mut leased = packet(0x37);
        let Payload::Exec(work) = &mut leased.payload else {
            panic!("an EXEC");
        };
        work.pipeline_leases.push(pipeline);
        let admitted = s.admit(&leased).expect("accepted");
        assert!(!admitted.ready, "a pipeline still compiling holds the work");
        assert_eq!(s.scheduler().waiting_on_pipelines(), 1);

        // A packet that is not GPU work leases nothing, so it waits for
        // nothing: only an EXEC binds a pipeline.
        assert!(s.admit(&packet(0x1e)).expect("accepted").ready);
        assert_eq!(s.scheduler().waiting_on_pipelines(), 1);
    }

    /// A pipeline finishing releases the work that was held for it.
    ///
    /// The table knew the pipeline had become ready and the scheduler knew who
    /// was parked on it, and nothing carried one to the other — so a
    /// transaction admitted with a pipeline wait was admitted into a wait
    /// nothing could discharge, holding its channel's publication head and
    /// every completion word behind it.
    #[test]
    fn a_pipeline_finishing_releases_the_work_that_was_held_for_it() {
        let mut s = session();
        let pipeline = ResourceId {
            slot: ObjectListRef(9),
            generation: SlotGeneration(1),
        };
        let gen = s.generation();
        s.pipelines().declare(pipeline, gen);
        let mut leased = packet(0x37);
        let Payload::Exec(work) = &mut leased.payload else {
            panic!("an EXEC");
        };
        work.pipeline_leases.push(pipeline);
        let admitted = s.admit(&leased).expect("accepted");
        assert!(!admitted.ready);
        assert!(s.take_ready().is_empty());

        // The intermediate steps are the compiling layer's and release nothing.
        for step in [
            crate::pipeline::PipelineState::Translating,
            crate::pipeline::PipelineState::Compiling,
        ] {
            s.pipelines().advance(pipeline, step);
        }
        assert!(
            s.take_ready().is_empty(),
            "a pipeline is not ready until it is"
        );

        assert!(s.pipeline_ready(pipeline));
        assert_eq!(s.take_ready(), vec![admitted.transaction.identity.ingress]);
        assert_eq!(s.scheduler().waiting_on_pipelines(), 0);

        // A second arrival of the same news is not a legal step and releases
        // nothing, which is what stops a late compile callback resurrecting a
        // pipeline the guest deleted.
        assert!(!s.pipeline_ready(pipeline));
    }

    /// A pipeline that will never build is refused at ingress once it is known,
    /// and the work already admitted for it is named rather than dropped.
    ///
    /// Named because the two outcomes a caller must not take are worse: made
    /// ready, the work executes against a pipeline that does not exist; dropped,
    /// it holds its channel's publication head forever.
    #[test]
    fn a_pipeline_that_will_never_build_names_the_work_it_stranded() {
        let mut s = session();
        let pipeline = ResourceId {
            slot: ObjectListRef(9),
            generation: SlotGeneration(1),
        };
        let gen = s.generation();
        s.pipelines().declare(pipeline, gen);
        let mut leased = packet(0x37);
        let Payload::Exec(work) = &mut leased.payload else {
            panic!("an EXEC");
        };
        work.pipeline_leases.push(pipeline);
        let admitted = s.admit(&leased).expect("accepted");

        let reason = crate::pipeline::RefusalReason::CompilationFailed("out of registers");
        assert_eq!(
            s.pipeline_refused(pipeline, reason),
            vec![admitted.transaction.identity.ingress]
        );
        // And the next packet binding it is refused at ingress rather than
        // admitted into a wait that cannot resolve.
        let err = s.admit(&leased).expect_err("it can never run");
        assert_eq!(
            err,
            Refusal::PipelineUnusable(crate::pipeline::LeaseRefusal::Refused { pipeline, reason })
        );
        assert_eq!(err.slug(), "pipeline_compilation_failed");
    }

    /// Work binding a pipeline this session never declared is refused, and not
    /// as the same failure as one that could not be built.
    #[test]
    fn work_binding_an_undeclared_pipeline_is_refused_as_absent() {
        let mut s = session();
        let pipeline = ResourceId {
            slot: ObjectListRef(9),
            generation: SlotGeneration(1),
        };
        let mut leased = packet(0x37);
        let Payload::Exec(work) = &mut leased.payload else {
            panic!("an EXEC");
        };
        work.pipeline_leases.push(pipeline);
        let err = s.admit(&leased).expect_err("nothing declared it");
        assert_eq!(
            err,
            Refusal::PipelineUnusable(crate::pipeline::LeaseRefusal::Absent { pipeline })
        );
        assert_eq!(err.slug(), "pipeline_absent");
        // The refusal consumed no ordinal, like every other one here.
        let gen = s.generation();
        s.pipelines().declare(pipeline, gen);
        for step in [
            crate::pipeline::PipelineState::Translating,
            crate::pipeline::PipelineState::Compiling,
            crate::pipeline::PipelineState::Ready,
        ] {
            s.pipelines().advance(pipeline, step);
        }
        let ok = s.admit(&leased).expect("declared and built");
        assert_eq!(ok.transaction.identity.ingress, IngressOrdinal(1));
        assert!(ok.ready, "a pipeline already built is nothing to wait for");
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

    /// Withdrawing a transaction stops it ordering later work, not only its
    /// channel.
    ///
    /// A transaction holds a position in three planes and a withdrawal used to
    /// release one of them. The dependency graph kept its accesses live, so
    /// every later transaction touching that backing took a hazard wait on an
    /// ordinal nothing would ever complete; and the readiness service kept it
    /// pending, and that is the only thing that decrements a dependent's
    /// remaining hazards. So un-stalling a channel stalled every later
    /// transaction that shared a backing with the one taken out.
    #[test]
    fn a_withdrawn_transaction_stops_ordering_the_work_behind_it() {
        let mut s = session();
        let doomed = touching(packet(0x37), vec![whole(1, AccessMode::Write)]);
        let after = touching(packet(0x37), vec![whole(1, AccessMode::Write)]);

        let d = s.admit(&doomed).expect("accepted");
        let a = s.admit(&after).expect("accepted");
        assert!(!a.ready, "it waits on the one before it");
        assert_eq!(a.hazard_waits, vec![d.transaction.identity.ingress]);
        assert_eq!(s.take_ready(), vec![d.transaction.identity.ingress]);

        s.withdraw(d.transaction.identity.ingress);
        assert_eq!(
            s.take_ready(),
            vec![a.transaction.identity.ingress],
            "the hazard it held is released, not left on an ordinal nothing completes"
        );
        assert_eq!(s.scheduler().pending(), 1, "and it is no longer pending");

        // Its accesses stop ordering anything admitted later, too: a third
        // writer waits on the one still live and not on the one that left.
        let l = s
            .admit(&touching(packet(0x37), vec![whole(1, AccessMode::Write)]))
            .expect("accepted");
        assert_eq!(l.hazard_waits, vec![a.transaction.identity.ingress]);
    }

    /// A withdrawn transaction publishes no completion word of its own, and
    /// still releases the ones queued behind it.
    ///
    /// The work never ran, so a stamp published for it is a value the guest
    /// acts on. What the guest is owed is the typed reason, which is the
    /// caller's to name.
    #[test]
    fn a_withdrawal_publishes_what_was_behind_it_and_not_its_own_word() {
        let mut s = session();
        let mut doomed = packet(0x37);
        doomed.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(1),
        });
        let mut behind = packet(0x37);
        behind.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(2),
        });
        let d = s.admit(&doomed).expect("accepted");
        let b = s.admit(&behind).expect("accepted");
        assert!(s.complete(b.transaction.identity.ingress).is_empty());

        assert_eq!(
            s.withdraw(d.transaction.identity.ingress)
                .into_iter()
                .map(|r| r.stamp)
                .collect::<Vec<_>>(),
            vec![behind.completion],
            "only the word behind it"
        );
        assert_eq!(
            s.scheduler().published_value(StampSlot(0)),
            Some(StampValue(2))
        );
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

    /// A guest's own channel commands reach the channel set, end to end.
    ///
    /// Bytes, resolve, apply, admit — because the defect this closes was
    /// exactly a missing link in that chain: the operation resolved, the model
    /// held the channels, and nothing joined them, so a correct guest that
    /// defined a FIFO and used it got `ChannelNotOpen` on every packet.
    ///
    /// The bootstrap door is deliberately not used here. Only the root domain
    /// is opened by hand, which is what a real session does — the ring exists
    /// before the guest can name anything — and everything after that is the
    /// guest's own bytes.
    #[test]
    fn a_guests_channel_commands_open_and_end_the_domain_it_then_submits_on() {
        const DEFINE: u16 = 0x30;
        const FREE: u16 = 0x31;
        let mut s = SessionModel::new(SessionId(1));
        s.open_channel(ChannelId(0)).expect("the root ring");

        let domain = ChannelId(2).0.to_le_bytes();
        let define = crate::control::resolve(Channel::Root, DEFINE, &domain).expect("a definition");
        let free = crate::control::resolve(Channel::Root, FREE, &domain).expect("a free");

        // Before the definition is applied, the domain it names is not one.
        assert_eq!(
            s.admit(&packet(0x37)),
            Err(Refusal::ChannelNotOpen {
                channel: ChannelId(2)
            })
        );
        assert_eq!(s.apply_control(define), Ok(()));
        assert!(s.channel_open(ChannelId(2)));
        let admitted = s.admit(&packet(0x37)).expect("the domain is open now");

        // The free is refused while that packet still owes publication, and the
        // domain stays open — a refused transition changes nothing.
        assert_eq!(
            s.apply_control(free),
            Err(ControlRefusal::Free(RetireRefusal::LivePositions {
                outstanding: 1
            }))
        );
        assert!(s.channel_open(ChannelId(2)));

        s.complete(admitted.transaction.identity.ingress);
        assert_eq!(s.apply_control(free), Ok(()));
        assert_eq!(
            s.admit(&packet(0x37)),
            Err(Refusal::ChannelNotOpen {
                channel: ChannelId(2)
            }),
            "the lifetime the guest ended is over"
        );
    }

    /// Redefining a live domain is refused rather than resetting its
    /// publication order, and the refusal is the opening owner's own.
    #[test]
    fn a_second_definition_of_a_live_domain_is_refused() {
        let mut s = session();
        let domain = ChannelId(2).0.to_le_bytes();
        let define = crate::control::resolve(Channel::Root, 0x30, &domain).expect("a definition");
        assert_eq!(
            s.apply_control(define),
            Err(ControlRefusal::Open(Refusal::ChannelAlreadyOpen {
                channel: ChannelId(2)
            }))
        );
        assert_eq!(
            s.apply_control(define).unwrap_err().slug(),
            "ingress_channel_already_open"
        );
    }

    /// Every control operation that is not a channel transition leaves this
    /// model alone — which is the claim, not an omission.
    ///
    /// A display command's content belongs to the layer that has a display and
    /// an inert payload does nothing; neither touches ordering. The census is
    /// over the whole ledger so a control row that grows an ordering effect
    /// cannot be added without this failing.
    #[test]
    fn only_the_two_channel_commands_change_what_this_model_holds() {
        use reims_vgpu_protocol::packets::LEDGER;
        let mut applied = 0usize;
        for p in LEDGER {
            let Some(kind) = crate::control::ControlKind::of(p.channel, p.opcode) else {
                continue;
            };
            if kind.channel_transition().is_some() {
                continue;
            }
            let op =
                crate::control::resolve(p.channel, p.opcode, &[0u8; 8]).expect("a control packet");
            let mut s = session();
            let before = (s.channel_open(ChannelId(2)), s.channel_open(ChannelId(3)));
            assert_eq!(s.apply_control(op), Ok(()), "{}", kind.name());
            assert_eq!(
                (s.channel_open(ChannelId(2)), s.channel_open(ChannelId(3))),
                before,
                "{} moved a channel",
                kind.name()
            );
            applied += 1;
        }
        assert_eq!(
            applied, 21,
            "the ledger's control rows less the two channel commands"
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
        assert_eq!(
            died.epoch, after_reset.epoch,
            "the epoch that died is named"
        );
        assert!(died.stranded.is_empty(), "nothing was admitted");
        assert_eq!(
            s.generation(),
            after_reset.session,
            "the guest has not reset; it still names what it named"
        );
        assert_eq!(s.device_state(), DeviceState::Lost);

        let replacement = s.recreate_device().expect("lost, so replaceable");
        assert_ne!(replacement, died.epoch);
        assert_eq!(s.generation(), after_reset.session);
    }

    /// A device loss takes every transaction admitted into it out of all three
    /// planes, and names them.
    ///
    /// Nothing will complete them: the thing that would is what was lost. Left
    /// in place, each one holds its channel's publication head forever and its
    /// accesses keep ordering work admitted after the replacement device
    /// arrives. They are withdrawn here rather than named and left for a caller
    /// to remember, which is also the only place they *can* be withdrawn — the
    /// positions are this model's and a caller cannot enumerate them.
    #[test]
    fn a_device_loss_strands_the_work_admitted_into_it_and_names_it() {
        let mut s = session();
        let mut first = touching(packet(0x37), vec![whole(1, AccessMode::Write)]);
        first.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(1),
        });
        let mut second = touching(packet(0x37), vec![whole(1, AccessMode::Write)]);
        second.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(2),
        });
        let a = s.admit(&first).expect("accepted");
        let b = s.admit(&second).expect("accepted");
        assert_eq!(s.scheduler().pending(), 2);

        let loss = s.device_lost();
        assert_eq!(loss.epoch, s.epoch());
        assert_eq!(
            loss.stranded,
            vec![
                a.transaction.identity.ingress,
                b.transaction.identity.ingress
            ],
            "in ingress order, so a report reads in the order the guest sent them"
        );
        assert!(
            loss.released.is_empty(),
            "neither had completed, so no word was owed behind them"
        );
        assert_eq!(s.scheduler().pending(), 0);
        assert_eq!(
            s.scheduler().published_value(StampSlot(0)),
            None,
            "work that never ran publishes no word the guest could act on"
        );

        // And the replacement device starts clean: a writer of the same backing
        // waits on nothing.
        s.recreate_device().expect("lost");
        let next = s.admit(&first).expect("accepted");
        assert!(
            next.hazard_waits.is_empty(),
            "the dead epoch's accesses no longer order anything"
        );
        assert!(next.ready);
    }

    /// A completion the host delivered before the device died is still owed to
    /// the guest, and a loss releases it.
    ///
    /// It was queued behind a position that is now stranded. Dropping it with
    /// the stranded work would lose a completion that really happened.
    #[test]
    fn a_loss_releases_a_word_that_was_waiting_behind_the_work_it_stranded() {
        let mut s = session();
        let mut head = packet(0x37);
        head.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(1),
        });
        let mut behind = packet(0x37);
        behind.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(2),
        });
        let h = s.admit(&head).expect("accepted");
        let b = s.admit(&behind).expect("accepted");
        // The second finished; the first is still running when the device dies.
        assert!(s.complete(b.transaction.identity.ingress).is_empty());

        let loss = s.device_lost();
        assert_eq!(loss.stranded, vec![h.transaction.identity.ingress]);
        assert_eq!(
            loss.released
                .into_iter()
                .map(|r| r.stamp)
                .collect::<Vec<_>>(),
            vec![behind.completion]
        );
        assert_eq!(
            s.scheduler().published_value(StampSlot(0)),
            Some(StampValue(2))
        );
    }

    /// Work submitted between a loss and its replacement has nothing to run on
    /// and must be told so, not admitted into an incarnation that is gone.
    #[test]
    fn a_lost_device_refuses_admission_until_it_is_replaced() {
        let mut s = session();
        let epoch = s.device_lost().epoch;
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

        assert!(
            s.device_lost().stranded.is_empty(),
            "this test admitted nothing"
        );
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
        assert!(
            a.device_lost().stranded.is_empty(),
            "the other session's work is not this one's to strand"
        );
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
        let mut reader = touching(packet(0x37), vec![whole(1, AccessMode::Read)]);
        reader.session = after;
        let r = s.admit(&reader).expect("accepted");
        assert_eq!(r.transaction.identity.session, after);
        assert_eq!(r.hazard_waits, vec![w.transaction.identity.ingress]);
    }

    /// A packet read under a lifetime that has since closed is refused, and the
    /// refusal names both generations.
    ///
    /// A reset races the drain: a packet that left the ring before it and
    /// reaches ingress after names objects that no longer exist. Nothing else
    /// can tell — the guest's bytes carry no generation, and by the time this
    /// plane sees the packet its own generation has already moved — which is
    /// why the reader states the one it was holding.
    ///
    /// Not the same event as the reset itself: work already *admitted* is
    /// untouched, which is the test above.
    #[test]
    fn a_packet_read_before_a_reset_is_refused_after_it() {
        let mut s = session();
        let stale = packet(0x37);
        let closed = s.generation();
        let current = s.reset();
        let err = s.admit(&stale).expect_err("its lifetime is over");
        assert_eq!(
            err,
            Refusal::GenerationClosed {
                named: closed,
                current,
            }
        );
        assert_eq!(err.slug(), "ingress_generation_closed");
        // And it consumed nothing, like every other refusal here.
        let mut fresh = packet(0x37);
        fresh.session = current;
        assert_eq!(
            s.admit(&fresh)
                .expect("accepted")
                .transaction
                .identity
                .ingress,
            IngressOrdinal(1)
        );
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
