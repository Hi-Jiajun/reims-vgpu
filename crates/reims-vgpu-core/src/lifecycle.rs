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
    AccessIntent, AccessRefusal, AccessSource, BackingId, ByteRange, Participation,
    ParticipationExtent, ResourceKey,
};
use crate::content::{ContentLedger, Replica, Transfer};
use crate::heap::{self, HeapPlacement, Heaps, Retirement};
use crate::identity::{ChannelId, ObjectListRef, ResourceId, TaskId};
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
    MapMemory {
        task: TaskId,
        resource: ResourceId,
    },
    UnmapMemory {
        task: TaskId,
        resource: ResourceId,
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

    #[must_use]
    pub const fn task(&self) -> TaskId {
        match self {
            Self::DefineTask { task }
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
}

impl ResolveRefusal {
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            Self::NotAResourceList { .. } => "lifecycle_not_a_resource_list",
            Self::Payload(inner) => inner.slug(),
            Self::UnknownRef { .. } => "lifecycle_unknown_ref",
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
        let refs: Vec<u32> = cmd.records.iter().map(|r| r.object_id).collect();
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

/// Why a lifecycle operation did not happen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    NoSuchTask {
        task: TaskId,
    },
    /// A live task cannot be redefined. Silently replacing would orphan every
    /// object the previous definition owns.
    TaskExists {
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
            Self::TaskExists { .. } => "lifecycle_task_exists",
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

#[derive(Debug, Default)]
struct Task {
    namespace: Namespace,
    heaps: Heaps,
    resident: HashMap<ResourceId, Resident>,
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
    /// If the task does not exist, or a live heap already has the number.
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
            LifecycleOp::DefineTask { task } => self.define_task(*task),
            LifecycleOp::DeleteTask { task } => self.delete_task(*task),
            LifecycleOp::CreateResource {
                task,
                slot,
                storage,
            } => self.create_resource(*task, *slot, *storage),
            LifecycleOp::DeleteResource { task, resource } => {
                self.delete_resource(*task, *resource)
            }
            LifecycleOp::MapMemory { task, resource } => self.set_mapping(*task, *resource, true),
            LifecycleOp::UnmapMemory { task, resource } => {
                self.set_mapping(*task, *resource, false)
            }
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

    fn define_task(&mut self, task: TaskId) -> Result<Effects, Refusal> {
        if self.tasks.contains_key(&task) {
            return Err(Refusal::TaskExists { task });
        }
        self.tasks.insert(task, Task::default());
        Ok(Effects::default())
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

    fn set_mapping(
        &mut self,
        task: TaskId,
        resource: ResourceId,
        mapped: bool,
    ) -> Result<Effects, Refusal> {
        let t = self
            .tasks
            .get_mut(&task)
            .ok_or(Refusal::NoSuchTask { task })?;
        let outcome = if mapped {
            t.namespace.map(resource)
        } else {
            t.namespace.unmap(resource)
        };
        outcome.map_err(|refusal| Refusal::Namespace { task, refusal })?;
        Ok(Effects::default())
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
        apply_inert(&mut l, &LifecycleOp::DefineTask { task: TASK });
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
        apply_inert(&mut l, &LifecycleOp::DefineTask { task: TASK });
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
        apply_inert(&mut l, &LifecycleOp::DefineTask { task: TASK });
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
        apply_inert(&mut l, &LifecycleOp::DefineTask { task: TASK });
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

    #[test]
    fn a_live_task_cannot_be_redefined() {
        let (mut l, _) = with_one_resource(256);
        assert_eq!(
            l.apply(&LifecycleOp::DefineTask { task: TASK }),
            Err(Refusal::TaskExists { task: TASK }),
            "silently replacing would orphan every object it owns"
        );
    }

    /// The contract retires the backing *and* the resources that named it, so
    /// this is not a refusal when they exist.
    #[test]
    fn deleting_a_backing_retires_the_resources_that_named_it() {
        let mut l = Lifecycle::new();
        apply_inert(&mut l, &LifecycleOp::DefineTask { task: TASK });
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

    #[test]
    fn mapping_is_declared_once_and_taken_away_once() {
        let (mut l, id) = with_one_resource(256);
        let map = LifecycleOp::MapMemory {
            task: TASK,
            resource: id,
        };
        let unmap = LifecycleOp::UnmapMemory {
            task: TASK,
            resource: id,
        };
        apply_inert(&mut l, &map);
        assert!(matches!(l.apply(&map), Err(Refusal::Namespace { .. })));
        apply_inert(&mut l, &unmap);
        assert!(matches!(l.apply(&unmap), Err(Refusal::Namespace { .. })));
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
            Refusal::TaskExists { task: TASK },
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
        apply_inert(&mut l, &LifecycleOp::DefineTask { task: TASK });
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
        let mut out = Vec::new();
        out.extend_from_slice(&task.to_le_bytes());
        out.extend_from_slice(&u32::try_from(refs.len()).expect("small").to_le_bytes());
        for r in refs {
            out.extend_from_slice(&r.to_le_bytes());
            out.resize(out.len() + record_len as usize - 4, 0);
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
