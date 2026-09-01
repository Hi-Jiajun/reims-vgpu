//! The resource-lifecycle transaction: every packet in the class, and what it
//! does to the owners of names, storage and content.
//!
//! # Why the class needs a vocabulary of its own
//!
//! [`crate::transaction::PayloadClass::ResourceLifecycle`] is one of five
//! payload classes and the only one whose members do genuinely different
//! things: a delete retires a name, a synchronise owes a copy, a discard
//! *offers* to drop one. Collapsing them into "the lifecycle packet" and
//! branching on the opcode at the point of use is how the replaced
//! architecture ended up with a partial mechanism per command. So the class has
//! an exhaustive resolved vocabulary — [`LifecycleOp`] — and
//! [`LifecycleKind::of`] is checked against the closure ledger so that "every
//! lifecycle packet has an operation" is a test rather than a claim.
//!
//! # Each command's effect is the ledger's reading of it, and not more
//!
//! - `CmdInvalidateResources` says the device's cached view of the guest's
//!   pages is stale. That is not a discard of a copy: it means the guest's
//!   pages are the current content and ours is not, which is exactly a write
//!   recorded in [`Replica::GuestPages`]. Recording it as a discard would leave
//!   the device replica able to claim freshness at the old version.
//! - `CmdSynchronizeResources` says the guest is about to read the pages with
//!   its CPU, so every write already submitted must have executed *before the
//!   packet completes*. It therefore owes copies, and those copies are the
//!   transaction's completion obligation rather than something a later read
//!   discovers.
//! - `CmdDiscardResources` releases a transfer copy that a later prepare or
//!   synchronise recreates. It is a hint: the ledger's reading is that ignoring
//!   it costs memory and not correctness. So a discard whose bytes exist in no
//!   other replica is *declined* and named, because taking it would not free a
//!   spare copy — it would destroy content.
//! - `CmdSynchronizeAndDiscardResources` is the two of them in that order, and
//!   the order is the whole point: after the synchronise the guest holds the
//!   bytes, so the discard has a second holder and is never the declined kind.
//!
//! # A discard happens at completion, because submission is not completion
//!
//! Both discard forms come back in [`Effects::at_completion`] rather than being
//! applied when the packet is admitted. A discard applied at admission would
//! mark the device replica stale while the copies that make that safe are still
//! owed, and a read admitted in between would be planned against freshness that
//! does not exist yet. [`Lifecycle::complete`] is where content authority
//! actually moves, and it re-asks the sole-authority question at that point
//! rather than trusting an answer computed earlier.
//!
//! # Nothing is half-applied
//!
//! The four content commands name a list of resources. Every name in the list
//! is resolved before any of them takes effect, so a list with one stale name
//! refuses whole. A caller that saw a refusal after three of five resources had
//! been invalidated would have no way to describe the state it was left in.

use crate::access::{
    AccessIntent, AccessRefusal, AccessSource, BackingId, ByteRange, GuestSpan, Participation,
    ParticipationExtent, ResourceKey,
};
use crate::content::{ContentLedger, Replica, Transfer};
use crate::heap::{self, HeapPlacement, Heaps, Retirement};
use crate::identity::{ChannelId, DirectoryFrame, ObjectListRef, ResourceId, TaskId};
use crate::namespace::{self, Namespace, Teardown};
use crate::transaction::{classify, PayloadClass};
use reims_vgpu_protocol::fifo;
use reims_vgpu_protocol::packets::Channel;
use std::collections::HashMap;

/// Which lifecycle command a packet is.
///
/// Exhaustive over the packet classes [`classify`] calls
/// [`PayloadClass::ResourceLifecycle`], which is what the totality test checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LifecycleKind {
    DefineTask,
    DeleteTask,
    SetObjectList,
    DeleteResource,
    MapMemory,
    UnmapMemory,
    ReplacePhysical,
    Invalidate,
    Synchronize,
    SynchronizeAndDiscard,
    Discard,
    DeleteBacking,
}

impl LifecycleKind {
    /// The lifecycle command a packet is, or `None` if the packet is not a
    /// lifecycle packet at all.
    #[must_use]
    pub fn of(channel: Channel, opcode: u16) -> Option<Self> {
        if classify(channel, opcode) != Some(PayloadClass::ResourceLifecycle) {
            return None;
        }
        use Channel::{Child, Root};
        Some(match (channel, opcode) {
            // Root and child are two views of one flat opcode space; the shared
            // numbers appear twice on purpose rather than through a
            // fallthrough, so adding one to a single channel is a visible edit.
            (Root | Child, 0x38) => Self::DefineTask,
            (Root | Child, 0x20) => Self::DeleteTask,
            (Root | Child, 0x33) => Self::SetObjectList,
            (Child, 0x25) => Self::DeleteResource,
            (Child, 0x39) => Self::MapMemory,
            (Child, 0x22) => Self::UnmapMemory,
            (Child, 0x3c) => Self::ReplacePhysical,
            (Child, 0x34) => Self::Invalidate,
            (Child, 0x35) => Self::Synchronize,
            (Child, 0x3e) => Self::SynchronizeAndDiscard,
            (Child, 0x3f) => Self::Discard,
            (Child, 0x36) => Self::DeleteBacking,
            _ => return None,
        })
    }

    /// The length of one record in this command's counted resource list, or
    /// `None` when the command carries no such list.
    ///
    /// **Four of the twelve carry one, and they do not all use the same record
    /// length.** `Invalidate` carries the guest's validity quad beside each
    /// object ref, so its record is eight bytes; the three that only name
    /// objects use four. Nothing else here has a list at all: a task
    /// definition, an object-list bind, a single delete, a map, an unmap, a
    /// re-point and a backing retirement each name one thing.
    ///
    /// The map lives here rather than in [`reims_vgpu_protocol::fifo`] because
    /// that module holds the layouts and deliberately holds no opcodes — a
    /// second table of the same numbers cannot be kept honest by anything in
    /// the toolchain. This is the one table, and it is keyed on the kind rather
    /// than on the opcode for the same reason: [`Self::of`] already turned the
    /// opcode into a kind.
    ///
    /// Before it, the choice of decoder was made at each call site in the
    /// device's drain — three arms picked the four-byte decoder and one picked
    /// the eight-byte one, and nothing could compare them.
    #[must_use]
    pub const fn resource_list_record_len(self) -> Option<u32> {
        match self {
            Self::Invalidate => Some(fifo::CHILD_INVALIDATE_RECORD_LEN),
            Self::Synchronize | Self::SynchronizeAndDiscard | Self::Discard => {
                Some(fifo::CHILD_SYNCHRONIZE_RECORD_LEN)
            }
            Self::DefineTask
            | Self::DeleteTask
            | Self::SetObjectList
            | Self::DeleteResource
            | Self::MapMemory
            | Self::UnmapMemory
            | Self::ReplacePhysical
            | Self::DeleteBacking => None,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DefineTask => "define_task",
            Self::DeleteTask => "delete_task",
            Self::SetObjectList => "set_object_list",
            Self::DeleteResource => "delete_resource",
            Self::MapMemory => "map_memory",
            Self::UnmapMemory => "unmap_memory",
            Self::ReplacePhysical => "replace_physical",
            Self::Invalidate => "invalidate",
            Self::Synchronize => "synchronize",
            Self::SynchronizeAndDiscard => "synchronize_and_discard",
            Self::Discard => "discard",
            Self::DeleteBacking => "delete_backing",
        }
    }
}

/// Where a resource's bytes live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    /// Guest pages the resource names by itself.
    Dedicated {
        backing: BackingId,
        extent: ByteRange,
    },
    /// A window of a heap, at an offset the guest chose. The resource does not
    /// own storage; see [`crate::heap`].
    Placed { heap: u64, offset: u64, length: u64 },
}

/// One resolved resource-lifecycle operation.
///
/// Every variant carries the task it acts in, because an object-list name is
/// task-local and a resolution that reached across tasks would find whatever
/// shared the integer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LifecycleOp {
    DefineTask {
        task: TaskId,
        /// Whether the guest registered this as its kernel task.
        ///
        /// Carried because the packet's first word is `(task_id << 1) |
        /// is_kernel_task` and dropping the low bit makes the kernel task —
        /// whose own id is `0` — indistinguishable from a user task in slot
        /// zero. They are not the same registration and their teardowns do not
        /// cost the same.
        kernel: bool,
        /// The page frame the task's page table is rooted at.
        ///
        /// **What makes a redefinition of a live task answerable.** A guest
        /// does redefine one, and the question that decides what survives is
        /// whether the root moved: a definition at the same root re-declares
        /// the space it already had, and one at a different root means every
        /// address this device resolved through the old table now translates
        /// somewhere else — including the object list itself, which reads back
        /// as zeros through the new one.
        ///
        /// The packet's fourth word, the address-space length, is **not** here.
        /// No decision in this crate is a function of it — the model holds
        /// nothing keyed by a guest address, which is the same reason
        /// [`Self::MapMemory`] is an obligation rather than a table — and a
        /// field nothing reads is one that quietly acquires a wrong meaning. It
        /// stays on [`reims_vgpu_protocol::fifo::DefineTaskCommand`] for the
        /// layer that walks the space.
        directory: DirectoryFrame,
    },
    DeleteTask {
        task: TaskId,
    },
    /// The object-list walk's per-object result: this slot now holds a
    /// resource with this storage.
    CreateResource {
        task: TaskId,
        slot: ObjectListRef,
        storage: Storage,
    },
    DeleteResource {
        task: TaskId,
        resource: ResourceId,
    },
    /// The guest has given an interval of this task's GPU virtual address
    /// space pages, and says so after it has already applied the change to its
    /// own page table.
    ///
    /// **It names an address interval and no object.** The packet carries a
    /// task, a 64-bit base and a 64-bit length, and the closure ledger reads
    /// its opcode as establishing *a task GPU-VA mapping over guest pages* —
    /// there is no object ref in it to resolve. A model that took it for a
    /// per-object mapping would be asserting something the wire never says,
    /// and the resource it named would be whichever one shared the low half of
    /// an address.
    MapMemory {
        task: TaskId,
        span: GuestSpan,
    },
    /// The same interval, taken away. Arrives after the guest has removed the
    /// translations, so the addresses in it no longer resolve.
    UnmapMemory {
        task: TaskId,
        span: GuestSpan,
    },
    /// Re-point a resource at different guest pages.
    ReplacePhysical {
        task: TaskId,
        resource: ResourceId,
        backing: BackingId,
        extent: ByteRange,
    },
    Invalidate {
        task: TaskId,
        resources: Vec<ResourceId>,
    },
    Synchronize {
        task: TaskId,
        resources: Vec<ResourceId>,
    },
    SynchronizeAndDiscard {
        task: TaskId,
        resources: Vec<ResourceId>,
    },
    Discard {
        task: TaskId,
        resources: Vec<ResourceId>,
    },
    /// Retire a backing and the resources that named it.
    DeleteBacking {
        task: TaskId,
        backing: BackingId,
    },
}

impl LifecycleOp {
    #[must_use]
    pub const fn kind(&self) -> LifecycleKind {
        match self {
            Self::DefineTask { .. } => LifecycleKind::DefineTask,
            Self::DeleteTask { .. } => LifecycleKind::DeleteTask,
            Self::CreateResource { .. } => LifecycleKind::SetObjectList,
            Self::DeleteResource { .. } => LifecycleKind::DeleteResource,
            Self::MapMemory { .. } => LifecycleKind::MapMemory,
            Self::UnmapMemory { .. } => LifecycleKind::UnmapMemory,
            Self::ReplacePhysical { .. } => LifecycleKind::ReplacePhysical,
            Self::Invalidate { .. } => LifecycleKind::Invalidate,
            Self::Synchronize { .. } => LifecycleKind::Synchronize,
            Self::SynchronizeAndDiscard { .. } => LifecycleKind::SynchronizeAndDiscard,
            Self::Discard { .. } => LifecycleKind::Discard,
            Self::DeleteBacking { .. } => LifecycleKind::DeleteBacking,
        }
    }

    /// The resources this operation names, in the order it names them.
    ///
    /// Empty is a claim about the *operation*, not about the transaction: a
    /// task teardown names no resource and still tears down everything in the
    /// task, and a map notice names an address interval. What it means is that
    /// the operation itself makes no per-resource statement, so nothing here
    /// constrains the transaction's access list.
    ///
    /// Non-empty is the opposite, and it is what
    /// [`crate::transaction::LifecyclePayload`] holds the envelope to: these
    /// resources *are* the operation's statement about what it touches, so an
    /// envelope naming a different set is a hazard edge built against the wrong
    /// memory.
    #[must_use]
    pub fn resources(&self) -> &[ResourceId] {
        match self {
            Self::Invalidate { resources, .. }
            | Self::Synchronize { resources, .. }
            | Self::SynchronizeAndDiscard { resources, .. }
            | Self::Discard { resources, .. } => resources,
            Self::DeleteResource { resource, .. } | Self::ReplacePhysical { resource, .. } => {
                std::slice::from_ref(resource)
            }
            // A task's own lifetime, an address interval, a slot declaration
            // and a backing retirement each name something that is not a
            // resource id.
            Self::DefineTask { .. }
            | Self::DeleteTask { .. }
            | Self::CreateResource { .. }
            | Self::MapMemory { .. }
            | Self::UnmapMemory { .. }
            | Self::DeleteBacking { .. } => &[],
        }
    }

    #[must_use]
    pub const fn task(&self) -> TaskId {
        match self {
            Self::DefineTask { task, .. }
            | Self::DeleteTask { task }
            | Self::CreateResource { task, .. }
            | Self::DeleteResource { task, .. }
            | Self::MapMemory { task, .. }
            | Self::UnmapMemory { task, .. }
            | Self::ReplacePhysical { task, .. }
            | Self::Invalidate { task, .. }
            | Self::Synchronize { task, .. }
            | Self::SynchronizeAndDiscard { task, .. }
            | Self::Discard { task, .. }
            | Self::DeleteBacking { task, .. } => *task,
        }
    }
}

/// A copy the device offered to drop and did not, with the reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Declined {
    pub resource: ResourceId,
    pub backing: BackingId,
    /// Bytes that exist in no other replica. Dropping them would not free a
    /// spare copy; it would destroy content the guest may still read.
    pub sole_authority_bytes: u64,
}

/// A content-authority change owed to a transaction's completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeferredDiscard {
    pub resource: ResourceId,
    pub backing: BackingId,
    pub bytes: ByteRange,
}

/// What a lifecycle transaction obliges and leaves behind.
#[derive(Clone, Debug, Default, PartialEq)]
#[must_use = "the transfers a synchronise owes and the storage a delete freed are both obligations"]
pub struct Effects {
    /// Copies that must have executed before the transaction's completion
    /// stamp may publish.
    pub transfers: Vec<Transfer>,
    /// What the namespace said about each backing detached by this
    /// transaction.
    pub teardowns: Vec<Teardown>,
    /// Heap storage whose last allocation went with this transaction.
    pub storage_freed: Vec<BackingId>,
    /// Copies offered for release, evaluated when the transfers above have
    /// executed. See the module docs for why this is not immediate.
    pub at_completion: Vec<DeferredDiscard>,
    /// Address intervals whose translations the guest has changed under this
    /// device.
    pub remapped: Vec<Remap>,
    /// Task address spaces this transaction replaced under a live id.
    pub redefined: Vec<Redefinition>,
}

/// A live task redefined: the whole address space replaced under one id.
///
/// An obligation of the same kind [`Remap`] is, and a wider one. A remap moves
/// an interval; this replaces the table every interval was resolved through, so
/// *every* resolution held for the task is answered by the wrong pages
/// afterwards. The teardowns the redefinition performed travel in the same
/// [`Effects`] — nothing is orphaned — and this is what says the survivors'
/// cached addresses are not survivors.
///
/// `root_moved` is the distinction that decides how bad it is. A redefinition
/// at the same root re-declares the space the task already had, and what the
/// guest published into it is still there; one at a different root means the
/// object list itself is now a different page, and everything published into
/// the old one reads back as whatever the new pages hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Redefinition {
    pub task: TaskId,
    pub previous: DirectoryFrame,
    pub directory: DirectoryFrame,
    pub root_moved: bool,
}

/// An interval of a task's address space whose translations have changed.
///
/// An obligation and not a record: every resolution held over the interval was
/// computed against pages the guest has since moved, so a cache that keeps one
/// answers a later question with the wrong pages. This crate caches none, which
/// is exactly why the obligation has to leave it named.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Remap {
    pub task: TaskId,
    pub span: GuestSpan,
    /// Whether the interval has pages behind it now.
    ///
    /// It does not decide the invalidation — both directions invalidate, since
    /// the pages under a re-established mapping are new ones. It decides what
    /// may be read *after*: an unmapped interval has no translation at all, and
    /// a reader that treated the two alike would follow one the guest removed.
    pub established: bool,
}

/// Why a lifecycle packet's bytes did not become an operation.
///
/// Separate from [`Refusal`], which is why a well-formed operation could not be
/// *applied*. A payload that cannot be read and a task that does not exist are
/// different failures with different fixes, and folding them would make the log
/// unable to say which one happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveRefusal {
    /// The command carries no counted resource list, so there is nothing for
    /// [`resource_list`] to read. Not a malformed packet — a caller asking the
    /// wrong question.
    NotAResourceList { kind: LifecycleKind },
    /// The payload disagrees with itself: too short for its header, or
    /// declaring more records than it carries.
    Payload(fifo::ResourceListDecodeError),
    /// A ref in the list names no live object.
    ///
    /// Carries the guest's number, because the number is what a log line has to
    /// contain for anyone to find which object the guest thought it had.
    UnknownRef { object_ref: u32 },
    /// The command carries no `{task, object}` record, so there is nothing for
    /// [`object_reference`] to read.
    NotAnObjectReference { kind: LifecycleKind },
    /// The record resolved, and the operation still cannot be built: it names
    /// the pages behind the object, and a [`crate::resolve::RefResolver`]
    /// answers identity only.
    ///
    /// `ReplacePhysical` is the case. Its packet is a bare `{task, object}` —
    /// the guest re-points a resource at host frames it has already wired, at
    /// the *same* GPU-VA — so the new backing and extent are not on the wire at
    /// all. They come from whatever holds the object's storage, which is a
    /// registry this crate does not yet have. Named rather than approximated,
    /// because an operation carrying the *old* backing would re-point nothing
    /// while reporting success.
    NeedsStorage { kind: LifecycleKind },
    /// The command is not a backing retirement, so there is nothing for
    /// [`backing_retirement`] to read.
    NotABackingRetirement { kind: LifecycleKind },
    /// The retirement names a mapping the mapper holds no surface for.
    ///
    /// Carries the guest's number for [`Self::UnknownRef`]'s reason, and is a
    /// *different* refusal from it: the two numbers come from namespaces that
    /// overlap, so a log that spelled them the same way would send a reader to
    /// the object list to look for a mapping.
    UnknownMapping { mapping: u32 },
    /// The command is not a map-or-unmap notice, so there is no interval for
    /// [`map_notice`] to read. [`NotAResourceList`](Self::NotAResourceList)'s
    /// sibling, and a caller asking the wrong question rather than a malformed
    /// packet.
    NotAMapNotice { kind: LifecycleKind },
    /// The notice's payload cannot hold its three fields.
    ShortNotice(fifo::ShortPayload),
    /// The record resolved, and the operation is the result of walking a table
    /// in guest memory this crate cannot address.
    ///
    /// `SetObjectList` is the case. Its packet says where a task's object list
    /// is and how many entries it has; the operation is the per-entry result of
    /// reading it, and every byte of it lives in pages the memory bound owns.
    /// Named rather than approximated, for [`Self::NeedsStorage`]'s reason: an
    /// operation built from the packet alone would rebind nothing while
    /// reporting that it had.
    NeedsGuestTable { kind: LifecycleKind },
    /// An `Invalidate` record asks for a content-authority transition this
    /// device has no established meaning for.
    ///
    /// The record's four validity-op bytes are independent on the wire, and
    /// exactly one combination has been established: clear hostValid together
    /// with set guestValid, which is the guest saying it CPU-wrote the
    /// resource. Anything else is a transition no capture has shown, and the
    /// model has two honest answers — recover it, or refuse. Applying the
    /// guest-write transition regardless is the third answer and it is a wrong
    /// one: it moves authority the guest did not move, which is a stale draw or
    /// a lost CPU write with a clean log.
    ///
    /// Carries the guest's raw dword, because the whole point of the refusal is
    /// that someone reading the log can see which combination arrived.
    UnestablishedValidityOps { object_ref: u32, ops: u32 },
}

impl ResolveRefusal {
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            Self::NotAResourceList { .. } => "lifecycle_not_a_resource_list",
            Self::Payload(inner) => inner.slug(),
            Self::UnknownRef { .. } => "lifecycle_unknown_ref",
            Self::NotAnObjectReference { .. } => "lifecycle_not_an_object_reference",
            Self::NeedsStorage { .. } => "lifecycle_needs_storage",
            Self::NotABackingRetirement { .. } => "lifecycle_not_a_backing_retirement",
            Self::UnknownMapping { .. } => "lifecycle_unknown_mapping",
            Self::NotAMapNotice { .. } => "lifecycle_not_a_map_notice",
            Self::ShortNotice(_) => fifo::ShortPayload::SLUG,
            Self::NeedsGuestTable { .. } => "lifecycle_needs_guest_table",
            Self::UnestablishedValidityOps { .. } => "lifecycle_unestablished_validity_ops",
        }
    }
}

/// Turn one resource-list packet's payload into the operation it names.
///
/// **The link between "the wire has a resource list" and "the model has a
/// lifecycle operation".** [`reims_vgpu_protocol::fifo`] could read the list
/// and [`LifecycleOp`] could describe the result, and nothing joined them: the
/// only code that did was the device's drain, which acted on the decoded
/// records directly and produced no operation at all.
///
/// Every ref resolves or the whole packet refuses. A partial list is not an
/// option here — `Invalidate` says a set of resources went stale together, and
/// applying it to the subset that happened to resolve claims the others are
/// still fresh.
///
/// # Errors
///
/// [`ResolveRefusal`]: a kind with no list, a payload that disagrees with
/// itself, or a ref naming nothing live.
pub fn resource_list(
    kind: LifecycleKind,
    payload: &[u8],
    resolver: &impl crate::resolve::RefResolver,
) -> Result<LifecycleOp, ResolveRefusal> {
    let Some(record_len) = kind.resource_list_record_len() else {
        return Err(ResolveRefusal::NotAResourceList { kind });
    };
    // Which decoder is which record length, asked once. The eight-byte record
    // carries the guest's validity quad; the four-byte one is object refs
    // alone. `resource_list_record_len` is the only thing that decides.
    let (task, refs) = if record_len == fifo::CHILD_INVALIDATE_RECORD_LEN {
        let cmd = fifo::decode_invalidate_resources(payload).map_err(ResolveRefusal::Payload)?;
        // The quad beside each ref, checked rather than dropped. This is the
        // one list command whose records carry a transition of their own, and
        // the operation below states one transition for the whole packet — so
        // a record asking for a different one has to refuse here, or the
        // packet applies an authority move the guest did not ask for.
        let mut refs = Vec::with_capacity(cmd.records.len());
        for record in &cmd.records {
            if !record.ops.is_guest_write() {
                return Err(ResolveRefusal::UnestablishedValidityOps {
                    object_ref: record.object_id,
                    ops: record.flags,
                });
            }
            refs.push(record.object_id);
        }
        (cmd.task_id, refs)
    } else {
        let cmd = fifo::decode_synchronize_resources(payload).map_err(ResolveRefusal::Payload)?;
        (cmd.task_id, cmd.object_ids)
    };

    let mut resources = Vec::with_capacity(refs.len());
    for object_ref in refs {
        let resolved = resolver
            .resource(object_ref)
            .ok_or(ResolveRefusal::UnknownRef { object_ref })?;
        resources.push(resolved);
    }

    let task = TaskId(task);
    Ok(match kind {
        LifecycleKind::Invalidate => LifecycleOp::Invalidate { task, resources },
        LifecycleKind::Synchronize => LifecycleOp::Synchronize { task, resources },
        LifecycleKind::SynchronizeAndDiscard => {
            LifecycleOp::SynchronizeAndDiscard { task, resources }
        }
        LifecycleKind::Discard => LifecycleOp::Discard { task, resources },
        // Unreachable: `resource_list_record_len` answered `Some` for exactly
        // the four above, and a `None` returned before this point. Stated as a
        // refusal rather than a panic because the two functions are separate,
        // and a fifth kind gaining a record length without gaining an arm here
        // must be a caller-visible refusal rather than a lost packet.
        other => return Err(ResolveRefusal::NotAResourceList { kind: other }),
    })
}

/// The content-authority moves an EXEC packet declares in its own resource
/// table.
///
/// **A GPU-work packet carries a lifecycle statement, and it is the same
/// statement.** `EXEC_INDIRECT2` writes one 24-byte record per live object-list
/// entry, and the four bytes after each object ref are byte-identical to a
/// `CmdInvalidateResources` record's validity quad. The guest says "I CPU-wrote
/// this resource" here, with the submission that reads it, exactly as often as
/// it says it in a standalone packet — and if the two produced different
/// operations there would be two content-authority models, one of which is
/// reached only by whichever packet the guest happened to use.
///
/// So this returns the same [`LifecycleOp::Invalidate`], and the caller applies
/// it before the transaction's work reads anything. The task is the header's,
/// not a caller's guess: a table of object refs with someone else's namespace
/// resolves to someone else's resources.
///
/// **A zero quad is the normal record here and it is not a refusal.** Unlike a
/// standalone invalidate, whose whole purpose is to move authority, this table
/// lists every resource the submission touches — and most of them were not
/// CPU-written. A record asking for nothing contributes nothing, so an EXEC
/// that wrote none produces an operation over no resources.
///
/// # Errors
///
/// [`ResolveRefusal::ShortNotice`] for a payload too short for the header,
/// [`ResolveRefusal::Payload`] where the declared table is not there,
/// [`ResolveRefusal::UnestablishedValidityOps`] for a quad that is neither the
/// guest-write move nor nothing, and [`ResolveRefusal::UnknownRef`] for a ref
/// naming nothing live.
pub fn exec_resource_table(
    payload: &[u8],
    resolver: &impl crate::resolve::RefResolver,
) -> Result<LifecycleOp, ResolveRefusal> {
    let header = fifo::decode_exec_header(payload).map_err(ResolveRefusal::ShortNotice)?;
    let table = fifo::decode_exec_resource_table(payload).map_err(ResolveRefusal::Payload)?;
    let mut resources = Vec::new();
    for record in &table {
        if record.ops == fifo::InvalidateValidityOps::default() {
            continue;
        }
        if !record.ops.is_guest_write() {
            return Err(ResolveRefusal::UnestablishedValidityOps {
                object_ref: record.object_id,
                ops: record.ops.to_le_dword(),
            });
        }
        resources.push(
            resolver
                .resource(record.object_id)
                .ok_or(ResolveRefusal::UnknownRef {
                    object_ref: record.object_id,
                })?,
        );
    }
    Ok(LifecycleOp::Invalidate {
        task: TaskId(header.task_id),
        resources,
    })
}

/// Turn any lifecycle packet's payload into the operation it names.
///
/// **The one place a kind picks its join.** The five joins below each refuse
/// every kind that is not their own, which makes a wrong pairing a value rather
/// than a misread — but nothing said which one a given kind belongs to, so every
/// caller with a packet had to know, and a caller that knew wrongly would read
/// one command's offsets out of another's payload. `every_lifecycle_kind_
/// reaches_at_most_one_join` could see that no kind is read by two; it could not
/// make the choice for anyone.
///
/// The match is exhaustive over [`LifecycleKind`], so a thirteenth command is a
/// compile error here rather than a packet that quietly reaches whichever arm
/// came last.
///
/// Both resolvers are taken because the twelve commands need both namespaces
/// and they are different namespaces — an object-list ref and a mapping id
/// arrive as `u32`s that overlap numerically and name unrelated things, which is
/// why [`crate::resolve::RefResolver`] and
/// [`crate::resolve::MappingResolver`] are two traits.
///
/// # Errors
///
/// [`ResolveRefusal`] from the join that owns the kind, or — for the two
/// commands that resolve and still cannot become an operation —
/// [`ResolveRefusal::NeedsStorage`] and [`ResolveRefusal::NeedsGuestTable`],
/// which name what is missing rather than approximating it.
pub fn operation(
    kind: LifecycleKind,
    payload: &[u8],
    objects: &impl crate::resolve::RefResolver,
    mappings: &impl crate::resolve::MappingResolver,
) -> Result<LifecycleOp, ResolveRefusal> {
    match kind {
        LifecycleKind::DefineTask | LifecycleKind::DeleteTask => task_lifetime(kind, payload),
        LifecycleKind::DeleteResource | LifecycleKind::ReplacePhysical => {
            object_reference(kind, payload, objects)
        }
        LifecycleKind::MapMemory | LifecycleKind::UnmapMemory => map_notice(kind, payload),
        LifecycleKind::DeleteBacking => backing_retirement(kind, payload, mappings),
        LifecycleKind::Invalidate
        | LifecycleKind::Synchronize
        | LifecycleKind::SynchronizeAndDiscard
        | LifecycleKind::Discard => resource_list(kind, payload, objects),
        // The one command whose operation is not in its packet. See
        // `ResolveRefusal::NeedsGuestTable`.
        LifecycleKind::SetObjectList => Err(ResolveRefusal::NeedsGuestTable { kind }),
    }
}

/// Turn a task-lifetime packet's payload into the operation it names.
///
/// Reached through [`operation`], which is what decides that this is the join
/// a task-lifetime kind belongs to.
///
/// Takes no resolver, as [`map_notice`] does not: a task is named by its own
/// slot number and there is nothing about that number device state could answer
/// differently.
///
/// The two commands do **not** carry the id the same way. A definition's first
/// word is `(task_id << 1) | is_kernel_task`; a deletion's is a plain slot.
/// Reading either at the other's convention indexes every slot at twice or half
/// its number, which is why the shift lives in
/// [`reims_vgpu_protocol::fifo::DefineTaskId`] and not at a call site.
///
/// The definition's address-space extent and page-table directory are not
/// carried into the operation. They are the address-resolution owner's, this
/// crate resolves no address, and an operation holding a page frame it never
/// reads is a field that goes stale unnoticed.
///
/// # Errors
///
/// [`ResolveRefusal`]: a kind that is neither of the two, or a payload too
/// short. A short deletion is refused rather than defaulted, because slot `0`
/// is the kernel task.
pub fn task_lifetime(kind: LifecycleKind, payload: &[u8]) -> Result<LifecycleOp, ResolveRefusal> {
    match kind {
        LifecycleKind::DefineTask => {
            let command = fifo::decode_define_task(payload).map_err(ResolveRefusal::ShortNotice)?;
            Ok(LifecycleOp::DefineTask {
                task: TaskId(command.id.task_id),
                kernel: command.id.kernel,
                directory: DirectoryFrame(command.directory_pfn),
            })
        }
        LifecycleKind::DeleteTask => {
            let task = fifo::decode_delete_task(payload).map_err(ResolveRefusal::ShortNotice)?;
            Ok(LifecycleOp::DeleteTask { task: TaskId(task) })
        }
        other => Err(ResolveRefusal::NotAnObjectReference { kind: other }),
    }
}

/// Turn a `{task, object}` packet's payload into the operation it names.
///
/// Two commands carry that record. Only one of them becomes an operation from
/// it: a delete names what to retire and nothing else, while a re-point names
/// pages that are not on the wire — see [`ResolveRefusal::NeedsStorage`].
///
/// # Errors
///
/// [`ResolveRefusal`]: a kind carrying no such record, a payload too short, a
/// ref naming nothing live, or a kind whose operation needs storage this
/// resolver cannot name.
pub fn object_reference(
    kind: LifecycleKind,
    payload: &[u8],
    resolver: &impl crate::resolve::RefResolver,
) -> Result<LifecycleOp, ResolveRefusal> {
    if !matches!(
        kind,
        LifecycleKind::DeleteResource | LifecycleKind::ReplacePhysical
    ) {
        return Err(ResolveRefusal::NotAnObjectReference { kind });
    }
    let command = fifo::decode_task_object(payload).map_err(ResolveRefusal::ShortNotice)?;
    // The ref is resolved before the kind is judged, so a re-point naming a
    // dead object refuses as a dead object rather than as unfinished work here.
    let resource = resolver
        .resource(command.object_id)
        .ok_or(ResolveRefusal::UnknownRef {
            object_ref: command.object_id,
        })?;
    let task = TaskId(command.task_id);
    match kind {
        LifecycleKind::DeleteResource => Ok(LifecycleOp::DeleteResource { task, resource }),
        // `ReplacePhysical`, and nothing else reaches here.
        other => Err(ResolveRefusal::NeedsStorage { kind: other }),
    }
}

/// Turn a backing-retirement packet's payload into the operation it names.
///
/// The one join that needs a [`crate::resolve::MappingResolver`] rather than a
/// [`crate::resolve::RefResolver`]. Its record's first word is a **mapping**
/// id — the surface whose host backing is being retired — and mapping ids and
/// object-list refs are numerically overlapping namespaces for unrelated
/// things, so resolving this one through the object list would retire whatever
/// backing happened to share the integer.
///
/// The record is also the reverse of the `{task, object}` pair two other
/// commands carry: `{object, task}`. Nothing about the bytes distinguishes
/// them, which is why the offsets are
/// [`reims_vgpu_protocol::fifo::decode_delete_backing`]'s and not restated here.
///
/// # Errors
///
/// [`ResolveRefusal`]: a kind that is not a retirement, a payload too short, or
/// a mapping naming no live surface.
pub fn backing_retirement(
    kind: LifecycleKind,
    payload: &[u8],
    resolver: &impl crate::resolve::MappingResolver,
) -> Result<LifecycleOp, ResolveRefusal> {
    if kind != LifecycleKind::DeleteBacking {
        return Err(ResolveRefusal::NotABackingRetirement { kind });
    }
    let command = fifo::decode_delete_backing(payload).map_err(ResolveRefusal::ShortNotice)?;
    let mapping = crate::identity::MappingId(command.object_id);
    let backing = resolver
        .backing(mapping)
        .ok_or(ResolveRefusal::UnknownMapping {
            mapping: command.object_id,
        })?;
    Ok(LifecycleOp::DeleteBacking {
        task: TaskId(command.task_id),
        backing,
    })
}

/// Turn one map-or-unmap packet's payload into the operation it names.
///
/// **The join the resolver interface could not be the missing piece of.** Every
/// other lifecycle command that needed a byte-to-operation path needed a
/// [`crate::resolve::RefResolver`] to say which object a `u32` names; this pair
/// takes none, because the packet carries no ref. The interval it carries is
/// already total — a task, a base and a length — and there is nothing about it
/// that device state could answer differently.
///
/// # Errors
///
/// [`ResolveRefusal`]: a kind that is not one of the two, or a payload too
/// short to hold the interval. A short notice is refused rather than read with
/// a zero-filled tail: a length read short retires the wrong pages, and a base
/// read short names an address in the wrong half of the task's space.
pub fn map_notice(kind: LifecycleKind, payload: &[u8]) -> Result<LifecycleOp, ResolveRefusal> {
    if !matches!(kind, LifecycleKind::MapMemory | LifecycleKind::UnmapMemory) {
        return Err(ResolveRefusal::NotAMapNotice { kind });
    }
    let notice = fifo::decode_map_memory(payload).map_err(ResolveRefusal::ShortNotice)?;
    let task = TaskId(notice.task_id);
    let span = GuestSpan {
        base: notice.gva,
        length: notice.length,
    };
    Ok(match kind {
        LifecycleKind::UnmapMemory => LifecycleOp::UnmapMemory { task, span },
        // `MapMemory`, and every other kind returned above.
        _ => LifecycleOp::MapMemory { task, span },
    })
}

/// Why a lifecycle operation did not happen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    NoSuchTask {
        task: TaskId,
    },

    Namespace {
        task: TaskId,
        refusal: namespace::Refusal,
    },
    Heap {
        task: TaskId,
        refusal: heap::Refusal,
    },
    /// A heap-placed resource has no pages of its own, so there is nothing for
    /// a physical replacement to re-point. Re-pointing the heap's storage under
    /// it would move every other resource in the heap too.
    PlacedResourceHasNoPhysical {
        resource: ResourceId,
    },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NoSuchTask { .. } => "lifecycle_no_such_task",
            Self::Namespace { .. } => "lifecycle_namespace",
            Self::Heap { .. } => "lifecycle_heap",
            Self::PlacedResourceHasNoPhysical { .. } => "lifecycle_placed_resource_has_no_physical",
        }
    }
}

/// One task's records, in one submission domain, as an [`AccessSource`].
///
/// A borrow of the owner rather than a copy of anything out of it: the content
/// authority reserves versions while the transaction is being built, so a
/// source that held a snapshot would hand two records of one packet the same
/// reservation.
pub struct TaskAccess<'a> {
    lifecycle: &'a mut Lifecycle,
    task: TaskId,
    domain: ChannelId,
}

impl AccessSource for TaskAccess<'_> {
    fn access(&mut self, participation: &Participation) -> Result<AccessIntent, AccessRefusal> {
        self.lifecycle
            .access(self.task, self.domain, participation)
            .map_err(|refusal| AccessRefusal {
                resource: participation.resource,
                reason: refusal.slug(),
            })
    }
}

/// Where a live resource's bytes are, derived once at creation.
#[derive(Clone, Copy, Debug)]
enum Resident {
    Dedicated {
        backing: BackingId,
        extent: ByteRange,
    },
    Placed(HeapPlacement),
}

impl Resident {
    const fn backing(self) -> BackingId {
        match self {
            Self::Dedicated { backing, .. } => backing,
            Self::Placed(p) => p.backing,
        }
    }

    const fn extent(self) -> ByteRange {
        match self {
            Self::Dedicated { extent, .. } => extent,
            Self::Placed(p) => p.region,
        }
    }

    /// A window of the resource, in its backing's coordinates.
    ///
    /// Both storage shapes need the same check and neither may clamp: a
    /// resource placed in a heap has a neighbour one byte past its end, and a
    /// resource with its own pages has the next allocation there.
    fn window(self, offset: u64, length: u64) -> Result<ByteRange, heap::Refusal> {
        match self {
            Self::Placed(p) => p.within(offset, length),
            Self::Dedicated { extent, .. } => {
                if offset
                    .checked_add(length)
                    .is_none_or(|end| end > extent.length)
                {
                    return Err(heap::Refusal::OutOfPlacement {
                        offset,
                        length,
                        placement_length: extent.length,
                    });
                }
                Ok(ByteRange {
                    offset: extent.offset.saturating_add(offset),
                    length,
                })
            }
        }
    }
}

#[derive(Debug)]
struct Task {
    /// The page frame its page table is rooted at. See
    /// [`crate::identity::DirectoryFrame`].
    directory: DirectoryFrame,
    namespace: Namespace,
    heaps: Heaps,
    resident: HashMap<ResourceId, Resident>,
}

impl Task {
    fn new(directory: DirectoryFrame) -> Self {
        Self {
            directory,
            namespace: Namespace::new(),
            heaps: Heaps::default(),
            resident: HashMap::new(),
        }
    }
}

/// The resource-lifecycle owner for one session generation.
///
/// Holds the per-task name and heap owners, and the one session-wide content
/// authority: content is a property of a backing, and an IOSurface backing is
/// reachable from more than one task, so a per-task ledger would have two
/// answers to where the current bytes are.
#[derive(Debug, Default)]
pub struct Lifecycle {
    tasks: HashMap<TaskId, Task>,
    content: ContentLedger,
}

impl Lifecycle {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The session's content authority, for the executor that has to ask where
    /// bytes are without going through a lifecycle operation to do it.
    #[must_use]
    pub const fn content(&self) -> &ContentLedger {
        &self.content
    }

    /// The same authority, writable.
    ///
    /// **Not a second authority, and the distinction is what makes this
    /// legitimate.** A lifecycle operation's own content effects go through
    /// [`Self::apply`] and [`Self::complete`]; what this is for is the *other*
    /// event that changes where current bytes are — a transaction's writes
    /// landing at completion — which is not a lifecycle operation and has no
    /// operation to go through. Both events belong to one ledger, which is
    /// exactly the property a second ledger beside this one would break.
    ///
    /// [`crate::interpret::Interpreter`] is the caller: it holds this model and
    /// materialises the versions a completed transaction published. A caller
    /// that wanted to *declare* a backing before a run reaches the same way.
    pub const fn content_mut(&mut self) -> &mut ContentLedger {
        &mut self.content
    }

    /// Declare a heap into a task.
    ///
    /// Not a [`LifecycleOp`]: the ledger judges heap-backed *texture* creation
    /// and leaves heap destruction unresolved, so there is no established
    /// packet-level heap lifetime to give an operation to. This is the owner
    /// interface the creation route will use once that route is closed, and
    /// naming it here keeps [`Storage::Placed`] reachable without inventing a
    /// command.
    ///
    /// # Errors
    ///
    /// If the task does not exist, a live heap already has the number, or any
    /// heap of this task already holds the storage — see
    /// [`crate::heap::Refusal::StorageInUse`], which forwards through
    /// [`Refusal::Heap`].
    pub fn declare_heap(
        &mut self,
        task: TaskId,
        heap: u64,
        backing: BackingId,
        length: u64,
    ) -> Result<(), Refusal> {
        let t = self
            .tasks
            .get_mut(&task)
            .ok_or(Refusal::NoSuchTask { task })?;
        t.heaps
            .create(heap, backing, length)
            .map_err(|refusal| Refusal::Heap { task, refusal })
    }

    /// Resolve a task-local name to the backing and extent it currently holds.
    ///
    /// # Errors
    ///
    /// If the task or the name does not resolve.
    pub fn resolve(&mut self, task: TaskId, resource: ResourceId) -> Result<ByteRange, Refusal> {
        let t = self
            .tasks
            .get_mut(&task)
            .ok_or(Refusal::NoSuchTask { task })?;
        t.namespace
            .resolve(resource)
            .map_err(|refusal| Refusal::Namespace { task, refusal })?;
        Ok(t.resident.get(&resource).copied().map_or(
            ByteRange {
                offset: 0,
                length: 0,
            },
            Resident::extent,
        ))
    }

    /// This owner as an [`AccessSource`] for one task's records, in one
    /// submission domain.
    ///
    /// Both are properties of the *packet* and not of any one participation,
    /// which is why they are bound here rather than passed to
    /// [`Self::access`] by a caller that would have to repeat them per record.
    pub fn task_access(&mut self, task: TaskId, domain: ChannelId) -> TaskAccess<'_> {
        TaskAccess {
            lifecycle: self,
            task,
            domain,
        }
    }

    /// Place one participation: which backing it names, where in that
    /// backing's coordinates, and which content versions it consumes and
    /// produces.
    ///
    /// The single step from what a record said to what a scheduler can order,
    /// and it is here because this is the only owner that holds all three
    /// registries at once — the task's names, its heaps, and the session's
    /// content authority. Anywhere else it would be three lookups a caller
    /// could get out of step with each other.
    ///
    /// # The extent is translated, and a heap placement is not widened
    ///
    /// A record names a window of a *resource*; the dependency graph compares
    /// windows of a *backing*. For a resource with its own pages those are the
    /// same coordinates; for one placed in a heap they are not, and the single
    /// checked conversion is [`crate::heap::HeapPlacement::within`].
    ///
    /// A whole-resource participation is therefore two different keys. A
    /// dedicated resource *is* its backing, so it stays
    /// [`crate::access::AccessKey::Whole`] — the record named no range and the
    /// precision census should keep saying so. A placed resource is a window of
    /// a heap someone else's resource also sits in, so "the whole resource" is
    /// an exact range and naming the whole backing would alias every
    /// neighbour into it. The model knows that window exactly; claiming less
    /// would buy ordering the record never asked for.
    ///
    /// A subresource is not translated. Relating image coordinates to bytes
    /// needs a layout, which is an executor's — so it travels as the record
    /// wrote it and `may_alias` compares it against the backing.
    ///
    /// # Versions
    ///
    /// The input version is what is current over the memory named, and it is
    /// `None` where nothing has written it. A write reserves the next version
    /// now and commits it at completion, which is the reservation rule
    /// [`crate::coverage`] states: a reader planned against it waits for the
    /// work rather than for the plan.
    ///
    /// # Errors
    ///
    /// If the task or the name does not resolve, or the window leaves the
    /// resource. Nothing is reserved on a refusal.
    pub fn access(
        &mut self,
        task: TaskId,
        domain: ChannelId,
        participation: &Participation,
    ) -> Result<AccessIntent, Refusal> {
        let t = self
            .tasks
            .get_mut(&task)
            .ok_or(Refusal::NoSuchTask { task })?;
        t.namespace
            .resolve(participation.resource)
            .map_err(|refusal| Refusal::Namespace { task, refusal })?;
        let resident = *t
            .resident
            .get(&participation.resource)
            .ok_or(Refusal::Namespace {
                task,
                refusal: namespace::Refusal::NotDeclared {
                    slot: participation.resource.slot,
                },
            })?;
        let key = ResourceKey {
            backing: resident.backing(),
            heap: match resident {
                // The membership generation the participation is recorded
                // against, so a `useHeap` written before a neighbour arrived
                // still meets the resource that never moved — see
                // `HeapId::same_heap`.
                Resident::Placed(p) => t
                    .heaps
                    .membership(p.heap)
                    .map_err(|refusal| Refusal::Heap { task, refusal })
                    .map(Some)?,
                Resident::Dedicated { .. } => None,
            },
        };
        let extent = match participation.extent {
            ParticipationExtent::Range(range) => ParticipationExtent::Range(
                resident
                    .window(range.offset, range.length)
                    .map_err(|refusal| Refusal::Heap { task, refusal })?,
            ),
            ParticipationExtent::Subresource(range) => ParticipationExtent::Subresource(range),
            ParticipationExtent::Whole => match resident {
                Resident::Dedicated { .. } => ParticipationExtent::Whole,
                Resident::Placed(p) => ParticipationExtent::Range(p.region),
            },
        };
        let bytes = match extent {
            ParticipationExtent::Range(range) => Some(range),
            // The whole of a dedicated resource is the extent it was declared
            // with, which the content authority already knows.
            ParticipationExtent::Whole => self.content.extent(key.backing),
            ParticipationExtent::Subresource(_) => None,
        };
        let input = bytes.map_or_else(
            || self.content.newest_version(key.backing),
            |range| self.content.version_of(key.backing, range),
        );
        let output = participation
            .mode
            .writes()
            .then(|| self.content.reserve(key.backing));
        Ok(Participation {
            extent,
            ..*participation
        }
        .resolve(domain, key, input, output))
    }

    /// Apply one lifecycle operation.
    ///
    /// # Errors
    ///
    /// If the task, a name, or a heap does not resolve. A refused operation
    /// changes nothing, including a multi-resource one that refuses on its last
    /// name.
    pub fn apply(&mut self, op: &LifecycleOp) -> Result<Effects, Refusal> {
        match op {
            LifecycleOp::DefineTask {
                task, directory, ..
            } => self.define_task(*task, *directory),
            LifecycleOp::DeleteTask { task } => self.delete_task(*task),
            LifecycleOp::CreateResource {
                task,
                slot,
                storage,
            } => self.create_resource(*task, *slot, *storage),
            LifecycleOp::DeleteResource { task, resource } => {
                self.delete_resource(*task, *resource)
            }
            LifecycleOp::MapMemory { task, span } => self.remap(*task, *span, true),
            LifecycleOp::UnmapMemory { task, span } => self.remap(*task, *span, false),
            LifecycleOp::ReplacePhysical {
                task,
                resource,
                backing,
                extent,
            } => self.replace_physical(*task, *resource, *backing, *extent),
            LifecycleOp::Invalidate { task, resources } => self.invalidate(*task, resources),
            LifecycleOp::Synchronize { task, resources } => {
                self.synchronize(*task, resources, false)
            }
            LifecycleOp::SynchronizeAndDiscard { task, resources } => {
                self.synchronize(*task, resources, true)
            }
            LifecycleOp::Discard { task, resources } => self.discard(*task, resources),
            LifecycleOp::DeleteBacking { task, backing } => self.delete_backing(*task, *backing),
        }
    }

    /// Record that a transaction's transfers have executed, and take the
    /// content-authority changes it deferred to that point.
    ///
    /// Returns the discards that were offered and not taken. The sole-authority
    /// question is asked here rather than when the operation was admitted,
    /// because the transfers this transaction owed are exactly what may have
    /// changed the answer.
    pub fn complete(&mut self, effects: &Effects) -> Vec<Declined> {
        for transfer in &effects.transfers {
            self.content.record_transfer(transfer);
        }
        let mut declined = Vec::new();
        for d in &effects.at_completion {
            let sole = self
                .content
                .sole_authority(d.backing, d.bytes, Replica::DeviceOwned);
            if sole.is_empty() {
                self.content
                    .discard(d.backing, d.bytes, Replica::DeviceOwned);
            } else {
                declined.push(Declined {
                    resource: d.resource,
                    backing: d.backing,
                    sole_authority_bytes: sole.len(),
                });
            }
        }
        declined
    }

    /// Record that work wrote a window of a resource in one replica.
    ///
    /// The one place a resource-relative window becomes a backing-relative one
    /// for content purposes, and the reason it is here rather than at the
    /// executor: a heap-placed resource's neighbour begins one byte past its
    /// end, so an unchecked addition performed where the write is reported
    /// would claim freshness over content the writer never touched.
    ///
    /// # Errors
    ///
    /// If the task or the name does not resolve, or the window leaves the
    /// resource.
    pub fn record_write(
        &mut self,
        task: TaskId,
        resource: ResourceId,
        offset: u64,
        length: u64,
        replica: Replica,
    ) -> Result<(), Refusal> {
        let t = self
            .tasks
            .get_mut(&task)
            .ok_or(Refusal::NoSuchTask { task })?;
        t.namespace
            .resolve(resource)
            .map_err(|refusal| Refusal::Namespace { task, refusal })?;
        let resident = *t.resident.get(&resource).ok_or(Refusal::Namespace {
            task,
            refusal: namespace::Refusal::NotDeclared {
                slot: resource.slot,
            },
        })?;
        let bytes = resident
            .window(offset, length)
            .map_err(|refusal| Refusal::Heap { task, refusal })?;
        self.content.write(resident.backing(), bytes, replica);
        Ok(())
    }

    /// Install a task's address space, replacing one already under the id.
    ///
    /// **A live task is redefined, and refusing it loses the packet.** The
    /// previous position here was that silently replacing would orphan every
    /// object the old definition owns — which is true, and is an argument
    /// against replacing *silently* rather than against replacing. The guest
    /// does this: one macOS release never redefines a live task and a later one
    /// does, so a model that refuses it refuses an ordinary packet on the
    /// second.
    ///
    /// So the replacement goes through the same teardown a delete does, and
    /// nothing is orphaned: every resource retires by name, every heap that
    /// held the last allocation frees its storage, and all of it travels in the
    /// returned effects. What the caller additionally gets is a
    /// [`Redefinition`], because the objects are only half of it — every
    /// address resolved through the old page table is answered by the wrong
    /// pages now, and this crate holds no such resolution to invalidate.
    fn define_task(&mut self, task: TaskId, directory: DirectoryFrame) -> Result<Effects, Refusal> {
        let effects = match self.tasks.get(&task).map(|t| t.directory) {
            Some(previous) => {
                let mut effects = self.delete_task(task)?;
                effects.redefined.push(Redefinition {
                    task,
                    previous,
                    directory,
                    root_moved: previous != directory,
                });
                effects
            }
            None => Effects::default(),
        };
        self.tasks.insert(task, Task::new(directory));
        Ok(effects)
    }

    fn delete_task(&mut self, task: TaskId) -> Result<Effects, Refusal> {
        if !self.tasks.contains_key(&task) {
            return Err(Refusal::NoSuchTask { task });
        }
        let mut effects = Effects::default();
        for name in self.tasks[&task].namespace.live_names() {
            let one = self.delete_resource(task, name)?;
            effects.teardowns.extend(one.teardowns);
            effects.storage_freed.extend(one.storage_freed);
        }
        // Every allocation is gone, so each remaining heap frees its storage
        // now. A heap that still held one would have said so, and this would
        // be the leak that answer exists to prevent.
        let t = self.tasks.get_mut(&task).expect("checked above");
        for heap in t.heaps.live_heaps() {
            if let Ok(Retirement::StorageFree { backing }) = t.heaps.delete(heap) {
                effects.storage_freed.push(backing);
            }
        }
        for backing in &effects.storage_freed {
            self.content.forget(*backing);
        }
        self.tasks.remove(&task);
        Ok(effects)
    }

    fn create_resource(
        &mut self,
        task: TaskId,
        slot: ObjectListRef,
        storage: Storage,
    ) -> Result<Effects, Refusal> {
        let t = self
            .tasks
            .get_mut(&task)
            .ok_or(Refusal::NoSuchTask { task })?;
        // The heap window is checked before a name is published, so a refused
        // placement leaves no slot behind for the next declaration to trip on.
        let backing = match storage {
            Storage::Dedicated { backing, .. } => backing,
            Storage::Placed { heap, .. } => t
                .heaps
                .backing_of(heap)
                .map_err(|refusal| Refusal::Heap { task, refusal })?,
        };
        let id = t
            .namespace
            .declare(slot, backing)
            .map_err(|refusal| Refusal::Namespace { task, refusal })?;
        let resident = match storage {
            Storage::Dedicated { backing, extent } => {
                // Its own pages: the guest supplied them and holds all of the
                // content, so this is a declaration and not a write.
                self.content.declare(backing, extent, Replica::GuestPages);
                Resident::Dedicated { backing, extent }
            }
            Storage::Placed {
                heap,
                offset,
                length,
            } => {
                let placement = match t.heaps.place(heap, id, offset, length) {
                    Ok(p) => p,
                    Err(refusal) => {
                        // Undo the name. Its generation stays spent, which is
                        // the same thing a delete leaves behind and is what
                        // keeps a stale resolution from succeeding later.
                        let _ = t.namespace.delete(id);
                        return Err(Refusal::Heap { task, refusal });
                    }
                };
                // A window of a heap, so the authority is about those bytes
                // alone: declaring the whole backing would discard the
                // neighbours' content.
                self.content
                    .write(placement.backing, placement.region, Replica::GuestPages);
                Resident::Placed(placement)
            }
        };
        let t = self.tasks.get_mut(&task).expect("resolved above");
        t.resident.insert(id, resident);
        Ok(Effects::default())
    }

    fn delete_resource(&mut self, task: TaskId, resource: ResourceId) -> Result<Effects, Refusal> {
        let t = self
            .tasks
            .get_mut(&task)
            .ok_or(Refusal::NoSuchTask { task })?;
        let teardown = t
            .namespace
            .delete(resource)
            .map_err(|refusal| Refusal::Namespace { task, refusal })?;
        let resident = t.resident.remove(&resource);
        let mut effects = Effects {
            teardowns: vec![teardown],
            ..Effects::default()
        };
        match resident {
            Some(Resident::Placed(placement)) => {
                if let Ok(Retirement::StorageFree { backing }) = t.heaps.remove(placement, resource)
                {
                    effects.storage_freed.push(backing);
                    self.content.forget(backing);
                }
            }
            // Its own pages: nothing else names them, so the content goes with
            // the backing the namespace just handed back.
            Some(Resident::Dedicated { backing, .. }) => {
                if matches!(teardown, Teardown::Now { .. }) {
                    self.content.forget(backing);
                }
            }
            None => {}
        }
        Ok(effects)
    }

    /// Record that a task's translations over `span` have changed.
    ///
    /// This crate holds nothing keyed by a guest address, so there is nothing
    /// here to invalidate — the obligation belongs to whoever caches a
    /// resolution, and it is stated as an effect rather than performed. A live
    /// task is still required: an interval in a task that does not exist names
    /// no address space, and accepting it would let a stale channel retire
    /// another task's resolutions.
    fn remap(
        &mut self,
        task: TaskId,
        span: GuestSpan,
        established: bool,
    ) -> Result<Effects, Refusal> {
        if !self.tasks.contains_key(&task) {
            return Err(Refusal::NoSuchTask { task });
        }
        Ok(Effects {
            remapped: vec![Remap {
                task,
                span,
                established,
            }],
            ..Effects::default()
        })
    }

    fn replace_physical(
        &mut self,
        task: TaskId,
        resource: ResourceId,
        backing: BackingId,
        extent: ByteRange,
    ) -> Result<Effects, Refusal> {
        let t = self
            .tasks
            .get_mut(&task)
            .ok_or(Refusal::NoSuchTask { task })?;
        if matches!(t.resident.get(&resource), Some(Resident::Placed(_))) {
            return Err(Refusal::PlacedResourceHasNoPhysical { resource });
        }
        let teardown = t
            .namespace
            .replace_physical(resource, backing)
            .map_err(|refusal| Refusal::Namespace { task, refusal })?;
        t.resident
            .insert(resource, Resident::Dedicated { backing, extent });
        // The new pages are the guest's and every copy of the old ones is of
        // memory this resource no longer names.
        self.content.declare(backing, extent, Replica::GuestPages);
        if let Teardown::Now { backing: old } = teardown {
            self.content.forget(old);
        }
        Ok(Effects {
            teardowns: vec![teardown],
            ..Effects::default()
        })
    }

    fn invalidate(&mut self, task: TaskId, resources: &[ResourceId]) -> Result<Effects, Refusal> {
        for (_, resident) in self.resolve_all(task, resources)? {
            // The guest's pages are the current content and ours is not, which
            // is a write recorded in the guest's replica. Recorded as a discard
            // instead, the device replica could still claim freshness at the
            // old version.
            self.content
                .write(resident.backing(), resident.extent(), Replica::GuestPages);
        }
        Ok(Effects::default())
    }

    fn synchronize(
        &mut self,
        task: TaskId,
        resources: &[ResourceId],
        then_discard: bool,
    ) -> Result<Effects, Refusal> {
        let resolved = self.resolve_all(task, resources)?;
        let mut effects = Effects::default();
        for (resource, resident) in resolved {
            let (backing, extent) = (resident.backing(), resident.extent());
            if let Some(transfer) =
                self.content
                    .transfer_for_read(backing, extent, Replica::GuestPages)
            {
                effects.transfers.push(transfer);
            }
            if then_discard {
                effects.at_completion.push(DeferredDiscard {
                    resource,
                    backing,
                    bytes: extent,
                });
            }
        }
        Ok(effects)
    }

    fn discard(&mut self, task: TaskId, resources: &[ResourceId]) -> Result<Effects, Refusal> {
        let resolved = self.resolve_all(task, resources)?;
        Ok(Effects {
            at_completion: resolved
                .into_iter()
                .map(|(resource, resident)| DeferredDiscard {
                    resource,
                    backing: resident.backing(),
                    bytes: resident.extent(),
                })
                .collect(),
            ..Effects::default()
        })
    }

    fn delete_backing(&mut self, task: TaskId, backing: BackingId) -> Result<Effects, Refusal> {
        let t = self.tasks.get(&task).ok_or(Refusal::NoSuchTask { task })?;
        // The contract retires the backing *and* the resources that named it,
        // so this is not a refusal when they exist — it is a delete of each.
        let naming: Vec<ResourceId> = t
            .namespace
            .live_names()
            .into_iter()
            .filter(|id| t.resident.get(id).is_some_and(|r| r.backing() == backing))
            .collect();
        let mut effects = Effects::default();
        for resource in naming {
            let one = self.delete_resource(task, resource)?;
            effects.teardowns.extend(one.teardowns);
            effects.storage_freed.extend(one.storage_freed);
        }
        self.content.forget(backing);
        Ok(effects)
    }

    /// Resolve every name in a list before any of them takes effect.
    fn resolve_all(
        &mut self,
        task: TaskId,
        resources: &[ResourceId],
    ) -> Result<Vec<(ResourceId, Resident)>, Refusal> {
        let t = self
            .tasks
            .get_mut(&task)
            .ok_or(Refusal::NoSuchTask { task })?;
        let mut out = Vec::with_capacity(resources.len());
        for resource in resources {
            t.namespace
                .resolve(*resource)
                .map_err(|refusal| Refusal::Namespace { task, refusal })?;
            if let Some(resident) = t.resident.get(resource) {
                out.push((*resource, *resident));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::SlotGeneration;
    use reims_vgpu_protocol::packets::LEDGER;

    const TASK: TaskId = TaskId(1);

    fn range(offset: u64, length: u64) -> ByteRange {
        ByteRange { offset, length }
    }

    fn dedicated(backing: u64, length: u64) -> Storage {
        Storage::Dedicated {
            backing: BackingId(backing),
            extent: range(0, length),
        }
    }

    /// A task with one dedicated resource in slot 0, and the name it got.
    fn with_one_resource(length: u64) -> (Lifecycle, ResourceId) {
        let mut l = Lifecycle::new();
        apply_inert(
            &mut l,
            &LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            },
        );
        apply_inert(
            &mut l,
            &LifecycleOp::CreateResource {
                task: TASK,
                slot: ObjectListRef(0),
                storage: dedicated(10, length),
            },
        );
        (l, name(0))
    }

    /// Apply an operation that owes nothing, and say so rather than dropping
    /// the answer: `Effects` is `#[must_use]` because an ignored one is an
    /// obligation nobody holds.
    fn apply_inert(l: &mut Lifecycle, op: &LifecycleOp) {
        assert_eq!(l.apply(op).expect("resolves"), Effects::default());
    }

    fn name(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration::default().next(),
        }
    }

    /// The claim the module docs make and cannot check by being read: the
    /// vocabulary is total over the payload class.
    #[test]
    fn every_lifecycle_packet_has_exactly_one_operation() {
        let mut seen: Vec<LifecycleKind> = Vec::new();
        for p in LEDGER {
            let kind = LifecycleKind::of(p.channel, p.opcode);
            let is_lifecycle =
                classify(p.channel, p.opcode) == Some(PayloadClass::ResourceLifecycle);
            assert_eq!(
                kind.is_some(),
                is_lifecycle,
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
        assert_eq!(
            seen.len(),
            12,
            "every kind is a packet the ledger judged, and every judged \
             lifecycle packet is a kind"
        );
    }

    #[test]
    fn a_packet_that_is_not_lifecycle_has_no_kind() {
        assert_eq!(LifecycleKind::of(Channel::Child, 0x37), None, "the EXEC");
        assert_eq!(LifecycleKind::of(Channel::Child, 0x06), None, "a present");
        assert_eq!(LifecycleKind::of(Channel::Root, 0x00), None, "unjudged");
    }

    /// An invalidate says the device's cached view is stale. Recorded as a
    /// discard of a copy it would leave the device replica able to claim
    /// freshness at the old version; recorded as a guest write it cannot.
    #[test]
    fn an_invalidate_makes_the_guests_pages_the_current_content() {
        let (mut l, id) = with_one_resource(256);
        let sync = l
            .apply(&LifecycleOp::Synchronize {
                task: TASK,
                resources: vec![id],
            })
            .expect("resolves");
        assert!(sync.transfers.is_empty(), "the guest already holds them");
        // The device produced the whole resource, so it is the only holder.
        let backing = BackingId(10);
        l.record_write(TASK, id, 0, 256, Replica::DeviceOwned)
            .expect("inside the resource");
        assert!(l
            .content()
            .is_fresh(backing, range(0, 256), Replica::DeviceOwned));
        let version_before = l.content().version_of(backing, range(0, 256));
        apply_inert(
            &mut l,
            &LifecycleOp::Invalidate {
                task: TASK,
                resources: vec![id],
            },
        );
        assert!(
            l.content().version_of(backing, range(0, 256)) != version_before,
            "the content changed under us, so the version moved"
        );
        assert!(l
            .content()
            .is_fresh(backing, range(0, 256), Replica::GuestPages));
        assert!(!l
            .content()
            .is_fresh(backing, range(0, 1), Replica::DeviceOwned));
    }

    /// A synchronise owes exactly the bytes the guest is behind on, once.
    #[test]
    fn a_synchronise_owes_the_missing_bytes_and_does_not_repeat() {
        let (mut l, id) = with_one_resource(256);
        let backing = BackingId(10);
        // The device produced the back half.
        l.record_write(TASK, id, 128, 128, Replica::DeviceOwned)
            .expect("inside the resource");
        let _ = backing;
        let effects = l
            .apply(&LifecycleOp::Synchronize {
                task: TASK,
                resources: vec![id],
            })
            .expect("resolves");
        assert_eq!(effects.transfers.len(), 1);
        assert_eq!(effects.transfers[0].bytes.ranges(), &[range(128, 128)]);
        assert_eq!(effects.transfers[0].to, Replica::GuestPages);
        assert!(
            l.complete(&effects).is_empty(),
            "a synchronise declines nothing"
        );
        let again = l
            .apply(&LifecycleOp::Synchronize {
                task: TASK,
                resources: vec![id],
            })
            .expect("resolves");
        assert!(
            again.transfers.is_empty(),
            "no write in between, so nothing moved and nothing is owed"
        );
    }

    /// The reading the ledger records: the discard is a hint, and taking it
    /// when the bytes exist nowhere else would destroy content rather than
    /// free a copy.
    #[test]
    fn a_discard_of_content_nothing_else_holds_is_declined_and_named() {
        let (mut l, id) = with_one_resource(256);
        let backing = BackingId(10);
        l.record_write(TASK, id, 0, 256, Replica::DeviceOwned)
            .expect("inside the resource");
        let effects = l
            .apply(&LifecycleOp::Discard {
                task: TASK,
                resources: vec![id],
            })
            .expect("resolves");
        assert!(
            l.content()
                .is_fresh(backing, range(0, 256), Replica::DeviceOwned),
            "admission does not move content authority"
        );
        assert_eq!(
            l.complete(&effects),
            vec![Declined {
                resource: id,
                backing,
                sole_authority_bytes: 256,
            }]
        );
        assert!(
            l.content()
                .is_fresh(backing, range(0, 256), Replica::DeviceOwned),
            "and the declined hint left the copy alone"
        );
    }

    /// The order in the combined packet is the whole point: the synchronise
    /// gives the bytes a second holder, so the discard is always lawful.
    #[test]
    fn a_synchronise_and_discard_is_never_the_declined_kind() {
        let (mut l, id) = with_one_resource(256);
        let backing = BackingId(10);
        l.record_write(TASK, id, 0, 256, Replica::DeviceOwned)
            .expect("inside the resource");
        let effects = l
            .apply(&LifecycleOp::SynchronizeAndDiscard {
                task: TASK,
                resources: vec![id],
            })
            .expect("resolves");
        assert_eq!(
            effects.transfers.len(),
            1,
            "the guest is behind on all of it"
        );
        assert_eq!(
            l.complete(&effects),
            vec![],
            "the transfer this transaction owed is what made the hint lawful"
        );
        assert!(
            !l.content()
                .is_fresh(backing, range(0, 1), Replica::DeviceOwned),
            "so the copy was released"
        );
        assert!(
            l.content()
                .is_fresh(backing, range(0, 256), Replica::GuestPages),
            "and the content survived in the guest's pages"
        );
    }

    /// The same claim from the other side: had the discard been applied when
    /// the packet was admitted, the transfer it owed would have been planned
    /// against freshness that no longer existed.
    #[test]
    fn a_deferred_discard_is_evaluated_against_the_state_its_transfers_left() {
        let (mut l, id) = with_one_resource(256);
        l.record_write(TASK, id, 0, 256, Replica::DeviceOwned)
            .expect("inside the resource");
        let effects = l
            .apply(&LifecycleOp::SynchronizeAndDiscard {
                task: TASK,
                resources: vec![id],
            })
            .expect("resolves");
        // Complete without recording the transfers this transaction owed: the
        // guest never got the bytes, so the discard is the destroying kind.
        let without_transfers = Effects {
            transfers: Vec::new(),
            ..effects
        };
        assert_eq!(
            l.complete(&without_transfers).len(),
            1,
            "the answer is asked at completion and not cached from admission"
        );
    }

    #[test]
    fn a_list_with_one_stale_name_changes_nothing() {
        let (mut l, id) = with_one_resource(256);
        let backing = BackingId(10);
        let version = l.content().version_of(backing, range(0, 256));
        let stale = ResourceId {
            slot: ObjectListRef(9),
            generation: SlotGeneration::default().next(),
        };
        let refusal = l
            .apply(&LifecycleOp::Invalidate {
                task: TASK,
                resources: vec![id, stale],
            })
            .expect_err("the second name never resolved");
        assert_eq!(
            refusal,
            Refusal::Namespace {
                task: TASK,
                refusal: namespace::Refusal::NotDeclared {
                    slot: ObjectListRef(9)
                },
            }
        );
        assert_eq!(
            l.content().version_of(backing, range(0, 256)),
            version,
            "the first resource was not invalidated on the way to the refusal"
        );
    }

    /// A heap-placed resource does not own pages, so there is nothing to
    /// re-point — and re-pointing the heap's storage would move its neighbours.
    #[test]
    fn a_placed_resource_refuses_a_physical_replacement() {
        let mut l = Lifecycle::new();
        apply_inert(
            &mut l,
            &LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            },
        );
        l.declare_heap(TASK, 3, BackingId(50), 4096).expect("heap");
        apply_inert(
            &mut l,
            &LifecycleOp::CreateResource {
                task: TASK,
                slot: ObjectListRef(0),
                storage: Storage::Placed {
                    heap: 3,
                    offset: 0,
                    length: 256,
                },
            },
        );
        assert_eq!(
            l.apply(&LifecycleOp::ReplacePhysical {
                task: TASK,
                resource: name(0),
                backing: BackingId(60),
                extent: range(0, 256),
            }),
            Err(Refusal::PlacedResourceHasNoPhysical { resource: name(0) })
        );
    }

    /// Two resources in one heap share a backing, so creating the second must
    /// not declare the backing's authority — that would discard the first's
    /// content.
    #[test]
    fn placing_a_second_resource_leaves_the_first_ones_content_alone() {
        let mut l = Lifecycle::new();
        apply_inert(
            &mut l,
            &LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            },
        );
        l.declare_heap(TASK, 3, BackingId(50), 4096).expect("heap");
        let place = |l: &mut Lifecycle, slot: u32, offset: u64| {
            apply_inert(
                l,
                &LifecycleOp::CreateResource {
                    task: TASK,
                    slot: ObjectListRef(slot),
                    storage: Storage::Placed {
                        heap: 3,
                        offset,
                        length: 256,
                    },
                },
            );
        };
        place(&mut l, 0, 0);
        l.record_write(TASK, name(0), 0, 256, Replica::DeviceOwned)
            .expect("inside the resource");
        place(&mut l, 1, 256);
        assert!(
            l.content()
                .is_fresh(BackingId(50), range(0, 256), Replica::DeviceOwned),
            "the neighbour's copy survived"
        );
        assert!(
            l.content()
                .is_fresh(BackingId(50), range(256, 256), Replica::GuestPages),
            "and the new window is the guest's"
        );
    }

    #[test]
    fn deleting_a_task_retires_its_objects_and_frees_its_heap_storage() {
        let mut l = Lifecycle::new();
        apply_inert(
            &mut l,
            &LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            },
        );
        l.declare_heap(TASK, 3, BackingId(50), 4096).expect("heap");
        apply_inert(
            &mut l,
            &LifecycleOp::CreateResource {
                task: TASK,
                slot: ObjectListRef(0),
                storage: Storage::Placed {
                    heap: 3,
                    offset: 0,
                    length: 256,
                },
            },
        );
        apply_inert(
            &mut l,
            &LifecycleOp::CreateResource {
                task: TASK,
                slot: ObjectListRef(1),
                storage: dedicated(10, 128),
            },
        );
        let effects = l
            .apply(&LifecycleOp::DeleteTask { task: TASK })
            .expect("live task");
        assert_eq!(effects.teardowns.len(), 2, "both objects were retired");
        assert_eq!(
            effects.storage_freed,
            vec![BackingId(50)],
            "the heap's last allocation went with the task"
        );
        assert_eq!(
            l.apply(&LifecycleOp::DeleteTask { task: TASK }),
            Err(Refusal::NoSuchTask { task: TASK })
        );
    }

    /// A live task is redefined by the guest, and the model replaces it rather
    /// than refusing the packet.
    ///
    /// Refusing was the previous answer, on the ground that replacing silently
    /// would orphan the objects the old definition owns. It does — so the
    /// replacement is the delete's own teardown path, and the orphans arrive as
    /// effects instead of being invented away.
    #[test]
    fn a_live_task_is_redefined_and_its_objects_retire_by_name() {
        let (mut l, id) = with_one_resource(256);
        let effects = l
            .apply(&LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x2000),
            })
            .expect("a guest redefines a live task");
        assert_eq!(
            effects.redefined,
            vec![Redefinition {
                task: TASK,
                previous: DirectoryFrame(0x1000),
                directory: DirectoryFrame(0x2000),
                root_moved: true,
            }]
        );
        assert!(
            !effects.teardowns.is_empty(),
            "the resource the old definition owned retired by name"
        );
        // The name is the previous space's and does not resolve in the new one.
        assert_eq!(
            l.resolve(TASK, id),
            Err(Refusal::Namespace {
                task: TASK,
                refusal: namespace::Refusal::NotDeclared { slot: id.slot },
            })
        );
    }

    /// A redefinition at the same root is still a redefinition — the objects go
    /// either way — and it says the root did not move, because what the guest
    /// published into that page is still in it.
    #[test]
    fn a_redefinition_at_the_same_root_says_so() {
        let (mut l, _) = with_one_resource(256);
        let effects = l
            .apply(&LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            })
            .expect("redefined");
        assert_eq!(
            effects.redefined,
            vec![Redefinition {
                task: TASK,
                previous: DirectoryFrame(0x1000),
                directory: DirectoryFrame(0x1000),
                root_moved: false,
            }]
        );
    }

    /// A first definition replaces nothing and says nothing about a space that
    /// was not there.
    #[test]
    fn a_first_definition_is_not_a_redefinition() {
        let mut l = Lifecycle::new();
        let effects = l
            .apply(&LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            })
            .expect("a fresh task");
        assert_eq!(effects, Effects::default());
    }

    /// The contract retires the backing *and* the resources that named it, so
    /// this is not a refusal when they exist.
    #[test]
    fn deleting_a_backing_retires_the_resources_that_named_it() {
        let mut l = Lifecycle::new();
        apply_inert(
            &mut l,
            &LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            },
        );
        for slot in 0..2 {
            apply_inert(
                &mut l,
                &LifecycleOp::CreateResource {
                    task: TASK,
                    slot: ObjectListRef(slot),
                    storage: dedicated(10, 128),
                },
            );
        }
        apply_inert(
            &mut l,
            &LifecycleOp::CreateResource {
                task: TASK,
                slot: ObjectListRef(2),
                storage: dedicated(11, 128),
            },
        );
        let effects = l
            .apply(&LifecycleOp::DeleteBacking {
                task: TASK,
                backing: BackingId(10),
            })
            .expect("live task");
        assert_eq!(effects.teardowns.len(), 2, "and only those two");
        assert_eq!(
            l.resolve(TASK, name(2)),
            Ok(range(0, 128)),
            "the resource on the other backing is untouched"
        );
        assert!(matches!(
            l.resolve(TASK, name(0)),
            Err(Refusal::Namespace { .. })
        ));
    }

    /// A retirement's first word is a mapping and its second is the task, and
    /// the mapping resolves through the mapper rather than the object list.
    ///
    /// The record is the reverse of the `{task, object}` pair, and the two
    /// numbers here are different so a swap cannot pass: read backwards this
    /// would retire task 4's number as a mapping.
    #[test]
    fn a_backing_retirement_names_a_mapping_first_and_a_task_second() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&9u32.to_le_bytes());
        payload.extend_from_slice(&4u32.to_le_bytes());
        assert_eq!(
            backing_retirement(LifecycleKind::DeleteBacking, &payload, &EveryMapping),
            Ok(LifecycleOp::DeleteBacking {
                task: TaskId(4),
                backing: BackingId(1_009),
            })
        );
    }

    /// A mapping the mapper holds no surface for refuses under its own name.
    ///
    /// Not `UnknownRef`: that one names an object-list ref, and the two number
    /// spaces overlap, so one slug for both would send a reader to the object
    /// list to look for a mapping.
    #[test]
    fn a_retirement_naming_no_live_surface_refuses_as_a_mapping() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&9u32.to_le_bytes());
        payload.extend_from_slice(&4u32.to_le_bytes());
        let refusal = backing_retirement(LifecycleKind::DeleteBacking, &payload, &NoMapping)
            .expect_err("no live surface");
        assert_eq!(refusal, ResolveRefusal::UnknownMapping { mapping: 9 });
        assert_ne!(
            refusal.slug(),
            ResolveRefusal::UnknownRef { object_ref: 9 }.slug()
        );
    }

    /// A definition's word is doubled and a deletion's is not, and the join
    /// reads each the way its own command carries it.
    ///
    /// The pair that makes it matter: a definition of user task 1 sends `2`,
    /// and a deletion of user task 1 sends `1`. Swap the conventions and the
    /// definition registers task 2 while the deletion retires the kernel task.
    #[test]
    fn a_task_definition_is_doubled_and_a_deletion_is_not() {
        let mut define = vec![0u8; fifo::DEFINE_TASK_LEN];
        define[..4].copy_from_slice(
            &fifo::DefineTaskId {
                task_id: 1,
                kernel: false,
            }
            .to_raw()
            .to_le_bytes(),
        );
        // The page-table root, at its own offset. A definition that read it
        // from anywhere else would name a page the guest did not.
        define[fifo::DEFINE_TASK_DIRECTORY_PFN..fifo::DEFINE_TASK_DIRECTORY_PFN + 4]
            .copy_from_slice(&0x1000u32.to_le_bytes());
        assert_eq!(
            task_lifetime(LifecycleKind::DefineTask, &define),
            Ok(LifecycleOp::DefineTask {
                task: TaskId(1),
                kernel: false,
                directory: DirectoryFrame(0x1000),
            })
        );
        assert_eq!(
            task_lifetime(LifecycleKind::DeleteTask, &1u32.to_le_bytes()),
            Ok(LifecycleOp::DeleteTask { task: TaskId(1) })
        );

        // And the registration a dropped class bit would hide: the kernel task
        // and user task zero are two registrations of slot zero.
        let mut kernel = vec![0u8; fifo::DEFINE_TASK_LEN];
        kernel[..4].copy_from_slice(
            &fifo::DefineTaskId {
                task_id: 0,
                kernel: true,
            }
            .to_raw()
            .to_le_bytes(),
        );
        assert_eq!(
            task_lifetime(LifecycleKind::DefineTask, &kernel),
            Ok(LifecycleOp::DefineTask {
                task: TaskId(0),
                kernel: true,
                directory: DirectoryFrame(0),
            })
        );
    }

    /// A deletion with no id refuses. It does not name slot `0`, which is the
    /// kernel task.
    #[test]
    fn a_short_task_deletion_does_not_name_the_kernel_task() {
        assert_eq!(
            task_lifetime(LifecycleKind::DeleteTask, &[0u8; 3]),
            Err(ResolveRefusal::ShortNotice(fifo::ShortPayload {
                plen: 3,
                need: fifo::DELETE_TASK_LEN,
            }))
        );
        assert_eq!(
            task_lifetime(LifecycleKind::DefineTask, &[0u8; fifo::DEFINE_TASK_LEN - 1]),
            Err(ResolveRefusal::ShortNotice(fifo::ShortPayload {
                plen: fifo::DEFINE_TASK_LEN - 1,
                need: fifo::DEFINE_TASK_LEN,
            }))
        );
    }

    /// A delete names an object and becomes an operation; a re-point names the
    /// same two words and cannot.
    ///
    /// The re-point's new backing and extent are nowhere on the wire — the
    /// guest wired the pages itself and re-committed them at the same GPU-VA —
    /// so the refusal names what is missing rather than letting an operation
    /// carrying the *old* backing report a re-point that moved nothing.
    #[test]
    fn a_delete_resolves_from_its_ref_and_a_repoint_needs_more_than_one() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&TASK.0.to_le_bytes());
        payload.extend_from_slice(&7u32.to_le_bytes());
        let resource = ResourceId {
            slot: ObjectListRef(7),
            generation: SlotGeneration(1),
        };
        assert_eq!(
            object_reference(LifecycleKind::DeleteResource, &payload, &Everything),
            Ok(LifecycleOp::DeleteResource {
                task: TASK,
                resource,
            })
        );
        assert_eq!(
            object_reference(LifecycleKind::ReplacePhysical, &payload, &Everything),
            Err(ResolveRefusal::NeedsStorage {
                kind: LifecycleKind::ReplacePhysical,
            })
        );
    }

    /// The real namespace resolves a delete, and stops resolving it once the
    /// guest has deleted it.
    ///
    /// Every other test here uses a stub that answers about everything or
    /// nothing. This one drives `operation` from `crate::namespace::Namespace`,
    /// which is what the trait is for: the resolution path had no implementation
    /// but stubs, so nothing checked that what the namespace actually holds is
    /// what a command resolves against.
    #[test]
    fn a_delete_resolves_against_the_namespace_that_declared_the_object() {
        let mut names = crate::namespace::Namespace::new();
        let id = names
            .declare(ObjectListRef(7), crate::access::BackingId(10))
            .expect("a free slot");
        let mut payload = Vec::new();
        payload.extend_from_slice(&TASK.0.to_le_bytes());
        payload.extend_from_slice(&7u32.to_le_bytes());

        assert_eq!(
            operation(
                LifecycleKind::DeleteResource,
                &payload,
                &names,
                &EveryMapping
            ),
            Ok(LifecycleOp::DeleteResource {
                task: TASK,
                resource: id,
            })
        );

        // The generation is the object's own, not a number this test chose: a
        // second declaration in the same slot is a different name, and work
        // still carrying the first one no longer resolves to it.
        assert_eq!(
            names.delete(id).expect("declared"),
            crate::namespace::Teardown::Now {
                backing: crate::access::BackingId(10)
            },
            "nothing here acquired the lease, so the backing is owed to nobody"
        );
        assert_eq!(
            operation(
                LifecycleKind::DeleteResource,
                &payload,
                &names,
                &EveryMapping
            ),
            Err(ResolveRefusal::UnknownRef { object_ref: 7 }),
            "a deleted slot stops resolving"
        );
        let again = names
            .declare(ObjectListRef(7), crate::access::BackingId(11))
            .expect("the slot is free");
        assert_ne!(again, id);
        assert_eq!(
            operation(
                LifecycleKind::DeleteResource,
                &payload,
                &names,
                &EveryMapping
            ),
            Ok(LifecycleOp::DeleteResource {
                task: TASK,
                resource: again,
            })
        );
    }

    /// A ref naming nothing live refuses on the ref, for both kinds — the
    /// object is judged before the operation is.
    #[test]
    fn an_object_reference_naming_nothing_refuses_on_the_ref() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&TASK.0.to_le_bytes());
        payload.extend_from_slice(&7u32.to_le_bytes());
        for kind in [
            LifecycleKind::DeleteResource,
            LifecycleKind::ReplacePhysical,
        ] {
            assert_eq!(
                object_reference(kind, &payload, &Nothing),
                Err(ResolveRefusal::UnknownRef { object_ref: 7 })
            );
        }
    }

    /// Each join refuses every kind that is not its own, and the kinds that
    /// still have no join are named.
    ///
    /// Two claims in one test because they are the same census. A kind read by
    /// two joins would have one command's offsets applied to another's payload;
    /// a kind read by none is unfinished work, and listing them here is what
    /// keeps "which lifecycle commands can still not become an operation" an
    /// answer rather than a search.
    #[test]
    fn every_lifecycle_kind_reaches_at_most_one_join() {
        // Every kind the packet ledger classifies is in the list, so a
        // thirteenth cannot skip the sweep below.
        let mut classified: Vec<LifecycleKind> = LEDGER
            .iter()
            .filter_map(|p| LifecycleKind::of(p.channel, p.opcode))
            .collect();
        classified.sort_unstable();
        classified.dedup();
        let mut named = ALL_KINDS.to_vec();
        named.sort_unstable();
        assert_eq!(classified, named);

        // Long enough for any of the records, so a refusal is about the kind
        // and never about the length.
        let payload = vec![0u8; 64];
        let mut unjoined = Vec::new();
        for kind in ALL_KINDS {
            let joins = [
                task_lifetime(kind, &payload).is_ok(),
                object_reference(kind, &payload, &Everything).is_ok(),
                map_notice(kind, &payload).is_ok(),
                backing_retirement(kind, &payload, &EveryMapping).is_ok(),
                resource_list(kind, &payload, &Everything).is_ok(),
            ];
            let reached = joins.iter().filter(|ok| **ok).count();
            assert!(
                reached <= 1,
                "{} is read by more than one join",
                kind.name()
            );
            if reached == 0 {
                unjoined.push(kind);
            }
        }
        assert_eq!(
            unjoined,
            vec![
                // Its packet says where a task's object list is and how long;
                // the operation is the per-entry walk's result, and the walk
                // reads guest memory this crate does not address.
                LifecycleKind::SetObjectList,
                // Its record resolves and its operation still needs storage the
                // wire does not carry — see `ResolveRefusal::NeedsStorage`.
                LifecycleKind::ReplacePhysical,
            ],
            "the lifecycle commands that still cannot become an operation"
        );

        // And the dispatcher agrees with the sweep, kind for kind. The sweep
        // above says no kind is read by two joins; this says which one each is
        // read by, which is the half no caller could answer for itself.
        for kind in ALL_KINDS {
            let direct = [
                task_lifetime(kind, &payload),
                object_reference(kind, &payload, &Everything),
                map_notice(kind, &payload),
                backing_retirement(kind, &payload, &EveryMapping),
                resource_list(kind, &payload, &Everything),
            ]
            .into_iter()
            .find(Result::is_ok);
            let through = operation(kind, &payload, &Everything, &EveryMapping);
            match direct {
                Some(op) => assert_eq!(through, op, "{}", kind.name()),
                // The two with no join name what is missing rather than
                // reaching a join that would read another command's offsets.
                None => assert!(
                    matches!(
                        through,
                        Err(ResolveRefusal::NeedsStorage { .. }
                            | ResolveRefusal::NeedsGuestTable { .. })
                    ),
                    "{} became {through:?}",
                    kind.name()
                ),
            }
        }
    }

    /// The notice's fields as the packet carries them, and the direction the
    /// opcode carries.
    ///
    /// The interval is the whole payload: a `u32` task and two `u64`s. Reading
    /// the base as a `u32` — the shape the model asserted when it called this
    /// command a per-object mapping — would put the length word inside the
    /// address, so the high half is non-zero here.
    #[test]
    fn a_map_notice_is_a_task_and_an_interval_and_a_direction() {
        let mut payload = vec![0u8; fifo::MAP_MEMORY_LEN];
        payload[fifo::MAP_MEMORY_TASK_ID..fifo::MAP_MEMORY_TASK_ID + 4]
            .copy_from_slice(&TASK.0.to_le_bytes());
        payload[fifo::MAP_MEMORY_GVA..fifo::MAP_MEMORY_GVA + 8]
            .copy_from_slice(&0x0000_7f12_3456_1000u64.to_le_bytes());
        payload[fifo::MAP_MEMORY_LENGTH..fifo::MAP_MEMORY_LENGTH + 8]
            .copy_from_slice(&0x01c3_e000u64.to_le_bytes());
        let span = GuestSpan {
            base: 0x0000_7f12_3456_1000,
            length: 0x01c3_e000,
        };
        assert_eq!(
            map_notice(LifecycleKind::MapMemory, &payload),
            Ok(LifecycleOp::MapMemory { task: TASK, span })
        );
        assert_eq!(
            map_notice(LifecycleKind::UnmapMemory, &payload),
            Ok(LifecycleOp::UnmapMemory { task: TASK, span }),
            "the same record; the opcode is the whole difference"
        );
    }

    /// Every other lifecycle kind refuses by name rather than reading this
    /// record's offsets out of a payload that is not one.
    #[test]
    fn only_the_two_map_opcodes_carry_a_notice() {
        let payload = vec![0u8; fifo::MAP_MEMORY_LEN];
        for kind in [
            LifecycleKind::DefineTask,
            LifecycleKind::DeleteResource,
            LifecycleKind::ReplacePhysical,
            LifecycleKind::Invalidate,
            LifecycleKind::DeleteBacking,
        ] {
            assert_eq!(
                map_notice(kind, &payload),
                Err(ResolveRefusal::NotAMapNotice { kind })
            );
        }
    }

    /// A notice too short to hold its interval refuses; it does not become an
    /// interval with a zero-filled tail.
    #[test]
    fn a_short_notice_is_refused_and_not_zero_filled() {
        let payload = vec![0u8; fifo::MAP_MEMORY_LEN - 1];
        let refusal = map_notice(LifecycleKind::UnmapMemory, &payload)
            .expect_err("one byte short of the interval");
        assert_eq!(
            refusal,
            ResolveRefusal::ShortNotice(fifo::ShortPayload {
                plen: fifo::MAP_MEMORY_LEN - 1,
                need: fifo::MAP_MEMORY_LEN,
            })
        );
        assert_eq!(refusal.slug(), fifo::ShortPayload::SLUG);
    }

    /// Both directions leave the same obligation, and it names the interval the
    /// guest changed.
    ///
    /// The model holds nothing keyed by a guest address, so there is nothing
    /// here to invalidate — which is exactly why the obligation has to leave
    /// the crate named. A version that quietly returned no effects would be
    /// indistinguishable from one that had already discharged it.
    #[test]
    fn a_remap_leaves_the_interval_as_an_obligation() {
        let (mut l, _) = with_one_resource(256);
        let span = GuestSpan {
            base: 0x7f00_0000,
            length: 0x4000,
        };
        for (op, established) in [
            (LifecycleOp::MapMemory { task: TASK, span }, true),
            (LifecycleOp::UnmapMemory { task: TASK, span }, false),
        ] {
            let effects = l.apply(&op).expect("a live task");
            assert_eq!(
                effects.remapped,
                vec![Remap {
                    task: TASK,
                    span,
                    established,
                }]
            );
            assert!(
                effects.transfers.is_empty() && effects.teardowns.is_empty(),
                "a translation change moves no bytes and retires no backing"
            );
        }
    }

    /// An interval in a task that does not exist names no address space.
    #[test]
    fn a_remap_of_a_dead_task_is_refused() {
        let (mut l, _) = with_one_resource(256);
        let task = TaskId(TASK.0 + 1);
        assert_eq!(
            l.apply(&LifecycleOp::MapMemory {
                task,
                span: GuestSpan {
                    base: 0x1000,
                    length: 0x1000,
                },
            }),
            Err(Refusal::NoSuchTask { task })
        );
    }

    /// Two intervals of one task's space overlap on an address, and a
    /// zero-length one names none.
    #[test]
    fn an_empty_interval_overlaps_nothing_including_itself() {
        let a = GuestSpan {
            base: 0x1000,
            length: 0x1000,
        };
        assert!(a.overlaps(GuestSpan {
            base: 0x1fff,
            length: 1
        }));
        assert!(!a.overlaps(GuestSpan {
            base: 0x2000,
            length: 1
        }));
        let empty = GuestSpan {
            base: 0x1000,
            length: 0,
        };
        assert!(!empty.overlaps(empty));
        assert!(!a.overlaps(empty));
    }

    #[test]
    fn a_physical_replacement_moves_the_content_authority_with_the_pages() {
        let (mut l, id) = with_one_resource(256);
        l.record_write(TASK, id, 0, 256, Replica::DeviceOwned)
            .expect("inside the resource");
        let effects = l
            .apply(&LifecycleOp::ReplacePhysical {
                task: TASK,
                resource: id,
                backing: BackingId(20),
                extent: range(0, 256),
            })
            .expect("resolves");
        assert_eq!(
            effects.teardowns,
            vec![Teardown::Now {
                backing: BackingId(10)
            }]
        );
        assert!(
            l.content()
                .is_fresh(BackingId(20), range(0, 256), Replica::GuestPages),
            "the new pages are the guest's"
        );
        assert_eq!(
            l.resolve(TASK, id),
            Ok(range(0, 256)),
            "and the name resolves to them"
        );
    }

    #[test]
    fn every_refusal_has_its_own_slug() {
        let all = [
            Refusal::NoSuchTask { task: TASK },
            Refusal::Namespace {
                task: TASK,
                refusal: namespace::Refusal::NotDeclared {
                    slot: ObjectListRef(0),
                },
            },
            Refusal::Heap {
                task: TASK,
                refusal: heap::Refusal::NoSuchHeap { heap: 0 },
            },
            Refusal::PlacedResourceHasNoPhysical { resource: name(0) },
        ];
        let mut slugs: Vec<&str> = all.iter().map(|r| r.slug()).collect();
        slugs.sort_unstable();
        let count = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), count);
    }
    /// The check `record_write` exists for. A heap-placed resource's
    /// neighbour begins one byte past its end.
    #[test]
    fn a_recorded_write_cannot_leave_its_resource() {
        let mut l = Lifecycle::new();
        apply_inert(
            &mut l,
            &LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            },
        );
        l.declare_heap(TASK, 3, BackingId(50), 4096).expect("heap");
        for (slot, offset) in [(0u32, 0u64), (1, 256)] {
            apply_inert(
                &mut l,
                &LifecycleOp::CreateResource {
                    task: TASK,
                    slot: ObjectListRef(slot),
                    storage: Storage::Placed {
                        heap: 3,
                        offset,
                        length: 256,
                    },
                },
            );
        }
        assert_eq!(
            l.record_write(TASK, name(0), 128, 256, Replica::DeviceOwned),
            Err(Refusal::Heap {
                task: TASK,
                refusal: heap::Refusal::OutOfPlacement {
                    offset: 128,
                    length: 256,
                    placement_length: 256,
                },
            }),
            "a write that ran into the neighbour is refused, not clamped"
        );
        assert!(
            l.content()
                .is_fresh(BackingId(50), range(256, 256), Replica::GuestPages),
            "and the neighbour still holds its own content"
        );
        // The same window inside the resource lands at the heap offset.
        l.record_write(TASK, name(0), 128, 128, Replica::DeviceOwned)
            .expect("inside the resource");
        assert!(l
            .content()
            .is_fresh(BackingId(50), range(128, 128), Replica::DeviceOwned));
        assert!(
            l.content()
                .is_fresh(BackingId(50), range(256, 256), Replica::GuestPages),
            "and stopped at its own end"
        );
    }

    // ------------------------------------------- bytes to a lifecycle op

    /// Every lifecycle kind, so a test over the set cannot silently miss one
    /// that is added later.
    ///
    /// Held against [`LifecycleKind::of`] by
    /// `every_lifecycle_kind_reaches_at_most_one_join`, which requires every
    /// kind the packet ledger classifies to appear here — so a thirteenth kind
    /// cannot quietly skip a join test.
    const ALL_KINDS: [LifecycleKind; 12] = [
        LifecycleKind::DefineTask,
        LifecycleKind::DeleteTask,
        LifecycleKind::SetObjectList,
        LifecycleKind::DeleteResource,
        LifecycleKind::MapMemory,
        LifecycleKind::UnmapMemory,
        LifecycleKind::ReplacePhysical,
        LifecycleKind::Invalidate,
        LifecycleKind::Synchronize,
        LifecycleKind::SynchronizeAndDiscard,
        LifecycleKind::Discard,
        LifecycleKind::DeleteBacking,
    ];

    /// A resolver that answers every ref, so a resolution test is about the
    /// list and not about whether the objects exist.
    struct Everything;

    impl crate::resolve::RefResolver for Everything {
        fn resource(&self, object_ref: u32) -> Option<ResourceId> {
            Some(ResourceId {
                slot: ObjectListRef(object_ref),
                generation: SlotGeneration(1),
            })
        }
    }

    /// A mapper that resolves every mapping, and one that resolves none.
    ///
    /// Deliberately answers a *different* backing than [`Everything`]'s slot
    /// numbers would suggest: a test that used one number for both namespaces
    /// would pass against a resolver that confused them.
    struct EveryMapping;

    impl crate::resolve::MappingResolver for EveryMapping {
        fn backing(&self, mapping: crate::identity::MappingId) -> Option<BackingId> {
            Some(BackingId(u64::from(mapping.0) + 1_000))
        }
    }

    struct NoMapping;

    impl crate::resolve::MappingResolver for NoMapping {
        fn backing(&self, _mapping: crate::identity::MappingId) -> Option<BackingId> {
            None
        }
    }

    /// A resolver whose slots are all empty.
    struct Nothing;

    impl crate::resolve::RefResolver for Nothing {
        fn resource(&self, _object_ref: u32) -> Option<ResourceId> {
            None
        }
    }

    /// A payload too short to hold even a resource-list header.
    fn alloc_short() -> Vec<u8> {
        vec![0u8; 3]
    }

    /// A resource-list payload: `{task, count}` then `count` records of
    /// `record_len`, with the object ref in the first word of each.
    fn list_bytes(task: u32, refs: &[u32], record_len: u32) -> Vec<u8> {
        list_bytes_with(task, refs, record_len, PAGEON_DWORD)
    }

    /// The guest-write quad, as `pageBacking` writes it.
    const PAGEON_DWORD: u32 = 0x0100_0001;

    /// [`list_bytes`], with the eight-byte record's validity quad stated.
    ///
    /// The quad is the record's, not the packet's, so a fixture that left it
    /// zero was feeding a transition the guest never asks for.
    fn list_bytes_with(task: u32, refs: &[u32], record_len: u32, ops: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&task.to_le_bytes());
        out.extend_from_slice(&u32::try_from(refs.len()).expect("small").to_le_bytes());
        for r in refs {
            out.extend_from_slice(&r.to_le_bytes());
            if record_len == 8 {
                out.extend_from_slice(&ops.to_le_bytes());
            } else {
                out.resize(out.len() + record_len as usize - 4, 0);
            }
        }
        out
    }

    /// Exactly four of the twelve lifecycle commands carry a counted resource
    /// list, and the two record lengths are not interchangeable: `Invalidate`
    /// puts the guest's validity quad beside each ref. Reading one command's
    /// list at the other's stride walks off the records.
    #[test]
    fn exactly_the_four_list_commands_have_a_record_length() {
        let with: Vec<LifecycleKind> = LEDGER
            .iter()
            .filter_map(|p| LifecycleKind::of(p.channel, p.opcode))
            .filter(|k| k.resource_list_record_len().is_some())
            .collect();
        assert_eq!(
            with,
            vec![
                LifecycleKind::Invalidate,
                LifecycleKind::Synchronize,
                LifecycleKind::SynchronizeAndDiscard,
                LifecycleKind::Discard,
            ]
        );
        assert_eq!(
            LifecycleKind::Invalidate.resource_list_record_len(),
            Some(8)
        );
        for kind in [
            LifecycleKind::Synchronize,
            LifecycleKind::SynchronizeAndDiscard,
            LifecycleKind::Discard,
        ] {
            assert_eq!(kind.resource_list_record_len(), Some(4), "{}", kind.name());
        }
    }

    /// The join this function exists to be: a guest's bytes become the
    /// operation the model names, with every ref resolved.
    #[test]
    fn a_resource_list_payload_becomes_the_operation_its_command_names() {
        for kind in [
            LifecycleKind::Invalidate,
            LifecycleKind::Synchronize,
            LifecycleKind::SynchronizeAndDiscard,
            LifecycleKind::Discard,
        ] {
            let stride = kind.resource_list_record_len().expect("a list command");
            let bytes = list_bytes(7, &[11, 12, 13], stride);
            let op = resource_list(kind, &bytes, &Everything).expect("well formed");
            assert_eq!(op.kind(), kind, "the op is the command's own");
            assert_eq!(op.task(), TaskId(7));
            let resources = match &op {
                LifecycleOp::Invalidate { resources, .. }
                | LifecycleOp::Synchronize { resources, .. }
                | LifecycleOp::SynchronizeAndDiscard { resources, .. }
                | LifecycleOp::Discard { resources, .. } => resources.clone(),
                other => panic!("{other:?} is not a list operation"),
            };
            assert_eq!(
                resources.iter().map(|r| r.slot.0).collect::<Vec<_>>(),
                vec![11, 12, 13],
                "{}",
                kind.name()
            );
        }
    }

    /// An `EXEC_INDIRECT2` payload: the three header words, then one 24-byte
    /// record per `(object_ref, quad)`.
    fn exec_bytes(task: u32, records: &[(u32, u32)]) -> Vec<u8> {
        use reims_vgpu_protocol::fifo::{
            CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN, CHILD_EXEC_RESOURCE_TAIL_LEN,
        };
        let mut out = Vec::new();
        out.extend_from_slice(&task.to_le_bytes());
        out.extend_from_slice(&u32::try_from(records.len()).expect("small").to_le_bytes());
        // The command-buffer count. Not this function's table, and stated so a
        // record is never read out of a word that was meant to be a count.
        out.extend_from_slice(&0u32.to_le_bytes());
        for (object_ref, ops) in records {
            out.extend_from_slice(&object_ref.to_le_bytes());
            out.extend_from_slice(&ops.to_le_bytes());
            out.resize(out.len() + CHILD_EXEC_RESOURCE_TAIL_LEN as usize, 0);
        }
        assert_eq!(
            out.len(),
            12 + records.len() * CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize
        );
        out
    }

    /// An EXEC packet's own resource table carries the same guest-write
    /// statement a standalone invalidate does, and it becomes the same
    /// operation.
    ///
    /// Two content-authority models — one for each packet the guest could have
    /// used to say the same thing — is the shape this avoids.
    #[test]
    fn an_exec_tables_guest_write_becomes_the_same_invalidate() {
        let bytes = exec_bytes(1, &[(11, PAGEON_DWORD), (12, PAGEON_DWORD)]);
        assert_eq!(
            exec_resource_table(&bytes, &Everything),
            Ok(LifecycleOp::Invalidate {
                task: TaskId(1),
                resources: vec![name(11), name(12)],
            })
        );
    }

    /// A record asking for no move is the normal record in this table: it lists
    /// every resource the submission touches, and most were not CPU-written.
    ///
    /// Refusing them, or treating them as guest writes, are both wrong in the
    /// same direction — one loses every EXEC, the other declares a device
    /// replica stale on every submission and charges the rebuild to the next
    /// reader.
    #[test]
    fn an_exec_record_asking_for_nothing_moves_nothing() {
        let bytes = exec_bytes(1, &[(11, 0), (12, PAGEON_DWORD), (13, 0)]);
        assert_eq!(
            exec_resource_table(&bytes, &Everything),
            Ok(LifecycleOp::Invalidate {
                task: TaskId(1),
                resources: vec![name(12)],
            })
        );
        // A table of nothing but used resources declares no move at all.
        let quiet = exec_bytes(1, &[(11, 0)]);
        assert_eq!(
            exec_resource_table(&quiet, &Everything),
            Ok(LifecycleOp::Invalidate {
                task: TaskId(1),
                resources: Vec::new(),
            })
        );
    }

    /// The same unestablished quad refuses here as in the standalone packet.
    #[test]
    fn an_exec_record_asking_for_an_unestablished_move_refuses() {
        let bytes = exec_bytes(1, &[(11, 0x0100_0101)]);
        assert_eq!(
            exec_resource_table(&bytes, &Everything),
            Err(ResolveRefusal::UnestablishedValidityOps {
                object_ref: 11,
                ops: 0x0100_0101,
            })
        );
    }

    /// The task is the header's. A table of refs resolved in another task's
    /// namespace names another task's resources.
    #[test]
    fn an_exec_tables_task_is_the_headers() {
        let bytes = exec_bytes(9, &[]);
        assert_eq!(
            exec_resource_table(&bytes, &Everything).map(|op| op.task()),
            Ok(TaskId(9))
        );
    }

    /// A payload too short for the header refuses on the header, and one whose
    /// declared table is not there refuses on the table.
    #[test]
    fn an_exec_payload_that_disagrees_with_itself_refuses_by_name() {
        assert!(matches!(
            exec_resource_table(&[0u8; 4], &Everything),
            Err(ResolveRefusal::ShortNotice(_))
        ));
        let mut bytes = exec_bytes(1, &[(11, PAGEON_DWORD)]);
        bytes.truncate(bytes.len() - 1);
        assert!(matches!(
            exec_resource_table(&bytes, &Everything),
            Err(ResolveRefusal::Payload(_))
        ));
    }

    /// The operation the table produces is one `Lifecycle` applies: the guest's
    /// pages become the current content of exactly the resource the record
    /// named.
    #[test]
    fn an_exec_tables_invalidate_applies_like_any_other() {
        let (mut l, id) = with_one_resource(256);
        let backing = BackingId(10);
        l.record_write(TASK, id, 0, 256, Replica::DeviceOwned)
            .expect("inside the resource");
        assert!(l
            .content()
            .is_fresh(backing, range(0, 256), Replica::DeviceOwned));

        let bytes = exec_bytes(TASK.0, &[(0, PAGEON_DWORD)]);
        let op = exec_resource_table(&bytes, &Everything).expect("well formed");
        assert_eq!(l.apply(&op).expect("resolves"), Effects::default());
        assert!(l
            .content()
            .is_fresh(backing, range(0, 256), Replica::GuestPages));
        assert!(!l
            .content()
            .is_fresh(backing, range(0, 1), Replica::DeviceOwned));
    }

    /// A record asking for a transition this device has not established refuses
    /// the packet, naming the ref and the quad that arrived.
    ///
    /// The four validity-op bytes were decoded and dropped: the operation
    /// states one transition for the whole packet and every record got it,
    /// whatever it asked for. A quad of zeros asks for no move at all and a
    /// packet of them would have marked every resource guest-written — a
    /// device replica declared stale that nothing had written.
    #[test]
    fn an_invalidate_record_asking_for_an_unestablished_move_refuses() {
        for ops in [0x0000_0000, 0x0000_0001, 0x0100_0000, 0x0001_0100] {
            let bytes = list_bytes_with(7, &[11], 8, ops);
            assert_eq!(
                resource_list(LifecycleKind::Invalidate, &bytes, &Everything),
                Err(ResolveRefusal::UnestablishedValidityOps {
                    object_ref: 11,
                    ops
                }),
                "{ops:#010x}"
            );
        }
    }

    /// One bad quad refuses the whole packet, for the reason one unresolvable
    /// ref does: the list says a set of resources moved together, and applying
    /// it to the ones that happened to be well formed claims the rest did not.
    #[test]
    fn one_unestablished_quad_refuses_the_whole_list() {
        let mut bytes = list_bytes_with(7, &[11, 12], 8, PAGEON_DWORD);
        let second_ops = bytes.len() - 4;
        bytes[second_ops..].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            resource_list(LifecycleKind::Invalidate, &bytes, &Everything),
            Err(ResolveRefusal::UnestablishedValidityOps {
                object_ref: 12,
                ops: 0
            })
        );
    }

    /// The refusal names the check that produced it.
    #[test]
    fn the_unestablished_quad_refusal_has_its_own_slug() {
        assert_eq!(
            ResolveRefusal::UnestablishedValidityOps {
                object_ref: 11,
                ops: 0
            }
            .slug(),
            "lifecycle_unestablished_validity_ops"
        );
    }

    /// The stride is the command's, and reading a list at the other command's
    /// stride is not a near miss — it walks off the records entirely.
    #[test]
    fn a_list_read_at_the_other_commands_stride_does_not_agree() {
        // Three eight-byte records, read as if they were four-byte ones.
        let bytes = list_bytes(7, &[11, 12, 13], 8);
        let wide =
            resource_list(LifecycleKind::Invalidate, &bytes, &Everything).expect("its own stride");
        let narrow =
            resource_list(LifecycleKind::Synchronize, &bytes, &Everything).expect("long enough");
        assert_ne!(
            format!("{wide:?}").replace("Invalidate", "X"),
            format!("{narrow:?}").replace("Synchronize", "X"),
            "the two strides must not read the same refs out of one payload"
        );
    }

    /// One unresolvable ref refuses the whole packet. `Invalidate` says a set
    /// of resources went stale together; applying it to the subset that
    /// happened to resolve claims the others are still fresh.
    #[test]
    fn one_unknown_ref_refuses_the_whole_list() {
        let bytes = list_bytes(7, &[11, 12], 4);
        assert_eq!(
            resource_list(LifecycleKind::Synchronize, &bytes, &Nothing),
            Err(ResolveRefusal::UnknownRef { object_ref: 11 })
        );
        assert_eq!(
            resource_list(LifecycleKind::Synchronize, &bytes, &Nothing)
                .expect_err("refused")
                .slug(),
            "lifecycle_unknown_ref"
        );
    }

    /// A payload that disagrees with itself refuses with the decoder's own
    /// reason, forwarded rather than restated.
    #[test]
    fn a_payload_that_cannot_be_read_forwards_the_decoders_reason() {
        let short = resource_list(LifecycleKind::Synchronize, &[0u8; 3], &Everything)
            .expect_err("no header");
        assert_eq!(
            short,
            ResolveRefusal::Payload(fifo::ResourceListDecodeError::ShortHeader { plen: 3 })
        );
        assert_eq!(short.slug(), "resource_list_short_header");

        // A count the payload cannot carry is the other half.
        let mut bytes = list_bytes(7, &[11], 4);
        bytes[4..8].copy_from_slice(&9u32.to_le_bytes());
        let truncated = resource_list(LifecycleKind::Synchronize, &bytes, &Everything)
            .expect_err("nine records in one record's worth of bytes");
        assert_eq!(truncated.slug(), "resource_list_truncated");
    }

    /// The eight commands that carry no list say so, rather than being read as
    /// an empty one — an empty `Invalidate` is a real packet and means nothing
    /// went stale, which is not what a `DefineTask` means.
    #[test]
    fn a_command_with_no_list_refuses_rather_than_reading_an_empty_one() {
        for kind in [
            LifecycleKind::DefineTask,
            LifecycleKind::DeleteTask,
            LifecycleKind::SetObjectList,
            LifecycleKind::DeleteResource,
            LifecycleKind::MapMemory,
            LifecycleKind::UnmapMemory,
            LifecycleKind::ReplacePhysical,
            LifecycleKind::DeleteBacking,
        ] {
            // Both a payload long enough to look like a list header and one
            // too short to be one. The reason has to be "this command has no
            // list" either way: a `DefineTask` refused for a short *list*
            // sends the reader looking for a list that was never there.
            for bytes in [list_bytes(7, &[], 4), alloc_short()] {
                assert_eq!(
                    resource_list(kind, &bytes, &Everything),
                    Err(ResolveRefusal::NotAResourceList { kind }),
                    "{} at {} bytes",
                    kind.name(),
                    bytes.len()
                );
            }
        }
        // And an empty list on a command that has one is a real operation.
        let empty = resource_list(LifecycleKind::Discard, &list_bytes(7, &[], 4), &Everything)
            .expect("an empty discard is well formed");
        assert_eq!(
            empty,
            LifecycleOp::Discard {
                task: TaskId(7),
                resources: Vec::new(),
            }
        );
    }
}
