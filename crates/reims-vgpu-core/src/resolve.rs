//! Turning a decoded record into a resolved operation.
//!
//! # This is the second of decode's two questions, and it is answered once
//!
//! `reims_vgpu_protocol::decode` answers *what did the guest write* and hands
//! back records with the guest's own `u32` refs still on them. This module
//! answers *which object does each ref name*, which is a question about device
//! state, and it is the only place that answers it. A decoder that resolved
//! refs would have to borrow live device state to read a record; a model that
//! resolved them a second time downstream could get a different answer than the
//! one the operation was planned against. The whole replacement exists partly
//! to delete that second shape.
//!
//! # A ref that names nothing is a refusal, not a zero
//!
//! [`ResourceId`] carries a generation, so a guest that deletes an object and
//! creates another in the same slot produces an id the old work no longer
//! matches. Resolution is where that check happens: a ref with no live object
//! refuses by name and the operation never enters the model. Substituting a
//! null id would put an operation into the stream that names something that
//! does not exist, and the dependency graph would then order real work against
//! it.
//!
//! # Counted lists go to the arena, not into the operation
//!
//! A barrier's resource list is guest-sized. The operation names a window of
//! the transaction's arena — two `u32` — so one arena serves every class that
//! carries a list and no operation is as large as the largest list anyone sent.
//! Appending is the only copy in the path, and it copies resolved ids rather
//! than guest bytes.

use crate::blit::{
    BlitOp, BlitOptions, BufferSpan, FillPattern, ImagePitch, Origin3, Size3, TexturePoint,
    TextureSpan,
};
use crate::exec::ExecArenas;
use crate::icb::{CommandRange, IcbOp};
use crate::identity::ResourceId;
use crate::resource_state::{ResourceStateOp, ResourceStateTarget, SliceLevel};
use crate::sync::{BarrierOp, BarrierTarget, EventOp, FenceOp, ResourceSpan};
use reims_vgpu_protocol::decode::blit::{
    BlitRecord, FillPattern as RecordFill, Origin as RecordOrigin, Size as RecordSize,
    TextureEndpoint,
};
use reims_vgpu_protocol::decode::icb::IcbRecord;
use reims_vgpu_protocol::decode::residency::ResidencyRecord;
use reims_vgpu_protocol::decode::resource_state::{RecordTarget, ResourceStateRecord};
use reims_vgpu_protocol::decode::sync::{BarrierRecord, EventRecord, FenceRecord, SyncRecord};
use reims_vgpu_protocol::decode::{DecodeRefusal, RefBind};

/// What a guest ref names right now.
///
/// One method, because that is the whole of what this layer may ask device
/// state. Anything more — a resource's size, its format, where its content is —
/// is a question the model answers from its own records, and a resolver that
/// could answer it would make resolution a second place those facts live.
pub trait RefResolver {
    /// The live resource a ref names, or `None` when the slot holds nothing.
    fn resource(&self, object_ref: u32) -> Option<ResourceId>;
}

/// Why a record could not become an operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveRefusal {
    /// The bytes never became a record.
    Decode(DecodeRefusal),
    /// A ref names no live object.
    ///
    /// Carries the guest's number, because the number is what a log line has to
    /// contain for anyone to find which object the guest thought it had.
    UnknownRef { object_ref: u32 },
    /// A counted list is longer than an arena window can name.
    ///
    /// A window is two `u32`, so a transaction cannot hold more than `u32::MAX`
    /// entries across all its lists. Refusing is the honest answer: a truncated
    /// list is a barrier that orders less than the guest asked for, which is
    /// worse than one that did not happen.
    ArenaOverflow { wanted: usize },
}

impl ResolveRefusal {
    /// The stable reason string for the failure channel.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::Decode(inner) => inner.reason(),
            Self::UnknownRef { .. } => "resolve_ref_names_no_object",
            Self::ArenaOverflow { .. } => "resolve_list_exceeds_arena_window",
        }
    }
}

impl From<DecodeRefusal> for ResolveRefusal {
    fn from(inner: DecodeRefusal) -> Self {
        Self::Decode(inner)
    }
}

/// Resolve one ref, refusing by name when it holds nothing.
fn one(resolver: &impl RefResolver, object_ref: u32) -> Result<ResourceId, ResolveRefusal> {
    resolver
        .resource(object_ref)
        .ok_or(ResolveRefusal::UnknownRef { object_ref })
}

/// Append a resolved list to the arena and name the window.
///
/// The whole list resolves before anything is appended. A list with a dead ref
/// in the middle would otherwise leave half of itself in the arena behind a
/// window nobody names — harmless today and exactly the kind of thing that
/// stops being harmless when an arena is reused.
fn append_refs(
    arenas: &mut ExecArenas,
    resolver: &impl RefResolver,
    refs: &[RefBind],
) -> Result<ResourceSpan, ResolveRefusal> {
    let start = u32::try_from(arenas.resources.len())
        .map_err(|_| ResolveRefusal::ArenaOverflow { wanted: refs.len() })?;
    let len = u32::try_from(refs.len())
        .map_err(|_| ResolveRefusal::ArenaOverflow { wanted: refs.len() })?;
    if start.checked_add(len).is_none() {
        return Err(ResolveRefusal::ArenaOverflow { wanted: refs.len() });
    }
    arenas.resources.reserve(refs.len());
    let mark = arenas.resources.len();
    for entry in refs {
        match one(resolver, entry.object_ref.get()) {
            Ok(id) => arenas.resources.push(id),
            Err(refusal) => {
                arenas.resources.truncate(mark);
                return Err(refusal);
            }
        }
    }
    Ok(ResourceSpan { start, len })
}

fn origin(o: RecordOrigin) -> Origin3 {
    Origin3 {
        x: o.x,
        y: o.y,
        z: o.z,
    }
}

fn size(s: RecordSize) -> Size3 {
    Size3 {
        width: s.width,
        height: s.height,
        depth: s.depth,
    }
}

fn endpoint(
    resolver: &impl RefResolver,
    e: TextureEndpoint,
) -> Result<TexturePoint, ResolveRefusal> {
    Ok(TexturePoint {
        texture: one(resolver, e.texture_ref)?,
        slice: e.slice,
        level: e.level,
        origin: origin(e.origin),
    })
}

/// Resolve a transfer record.
pub fn blit(record: &BlitRecord, resolver: &impl RefResolver) -> Result<BlitOp, ResolveRefusal> {
    Ok(match *record {
        BlitRecord::BufferToBuffer {
            source_ref,
            source_offset,
            dest_ref,
            dest_offset,
            size: bytes,
        } => BlitOp::BufferToBuffer {
            source: one(resolver, source_ref)?,
            source_offset,
            dest: one(resolver, dest_ref)?,
            dest_offset,
            size: bytes,
        },
        BlitRecord::BufferToTexture {
            source_ref,
            source_offset,
            bytes_per_row,
            bytes_per_image,
            size: extent,
            dest,
            options,
        } => BlitOp::BufferToTexture {
            source: one(resolver, source_ref)?,
            source_offset,
            source_pitch: ImagePitch {
                bytes_per_row,
                bytes_per_image,
            },
            size: size(extent),
            dest: endpoint(resolver, dest)?,
            options: BlitOptions(options),
        },
        BlitRecord::TextureToBuffer {
            source,
            size: extent,
            dest_ref,
            dest_offset,
            bytes_per_row,
            bytes_per_image,
            options,
        } => BlitOp::TextureToBuffer {
            source: endpoint(resolver, source)?,
            size: size(extent),
            dest: one(resolver, dest_ref)?,
            dest_offset,
            dest_pitch: ImagePitch {
                bytes_per_row,
                bytes_per_image,
            },
            options: BlitOptions(u32::from(options)),
        },
        BlitRecord::TextureRegion {
            source,
            dest,
            size: extent,
            options,
        } => BlitOp::TextureRegion {
            source: endpoint(resolver, source)?,
            dest: endpoint(resolver, dest)?,
            size: size(extent),
            options: BlitOptions(options),
        },
        BlitRecord::TextureSlices {
            source_ref,
            source_slice,
            source_level,
            dest_ref,
            dest_slice,
            dest_level,
            slice_count,
            level_count,
        } => BlitOp::TextureSlices {
            source: TextureSpan {
                texture: one(resolver, source_ref)?,
                base_slice: source_slice,
                base_level: source_level,
                slice_count,
                level_count,
            },
            dest: TextureSpan {
                texture: one(resolver, dest_ref)?,
                base_slice: dest_slice,
                base_level: dest_level,
                slice_count,
                level_count,
            },
        },
        BlitRecord::FillBuffer {
            buffer_ref,
            location,
            length,
            pattern,
        } => BlitOp::FillBuffer {
            dest: BufferSpan {
                buffer: one(resolver, buffer_ref)?,
                offset: location,
                length,
            },
            pattern: match pattern {
                RecordFill::Byte(b) => FillPattern::Byte(b),
                RecordFill::Pattern4(w) => FillPattern::Pattern4(w),
            },
        },
        BlitRecord::GenerateMipmaps { texture_ref } => BlitOp::GenerateMipmaps {
            texture: one(resolver, texture_ref)?,
        },
    })
}

/// Resolve a content-representation record.
pub fn resource_state(
    record: &ResourceStateRecord,
    resolver: &impl RefResolver,
) -> Result<ResourceStateOp, ResolveRefusal> {
    let target = match record.target {
        RecordTarget::WholeResource { object_ref } => ResourceStateTarget::Resource {
            resource: one(resolver, object_ref)?,
            subresource: None,
        },
        RecordTarget::SliceLevel {
            texture_ref,
            slice,
            level,
        } => ResourceStateTarget::Resource {
            resource: one(resolver, texture_ref)?,
            subresource: Some(SliceLevel { slice, level }),
        },
        RecordTarget::Encoder => ResourceStateTarget::Encoder,
    };
    Ok(ResourceStateOp {
        directive: record.directive,
        target,
    })
}

/// Resolve an indirect-command record.
pub fn icb(record: &IcbRecord, resolver: &impl RefResolver) -> Result<IcbOp, ResolveRefusal> {
    Ok(match *record {
        IcbRecord::ExecuteRange { icb_ref, commands } => IcbOp::ExecuteRange {
            icb: one(resolver, icb_ref)?,
            commands: CommandRange {
                location: commands.location,
                length: commands.length,
            },
        },
        IcbRecord::ExecuteIndirect { icb_ref, arguments } => IcbOp::ExecuteIndirect {
            icb: one(resolver, icb_ref)?,
            arguments: crate::bind::IndirectSource {
                buffer: one(resolver, arguments.buffer_ref)?,
                offset: arguments.offset,
            },
        },
        IcbRecord::Optimize { icb_ref, commands } => IcbOp::Optimize {
            icb: one(resolver, icb_ref)?,
            commands: CommandRange {
                location: commands.location,
                length: commands.length,
            },
        },
    })
}

/// Resolve a fence record.
pub fn fence(record: &FenceRecord, resolver: &impl RefResolver) -> Result<FenceOp, ResolveRefusal> {
    Ok(FenceOp {
        kind: record.kind,
        fence: one(resolver, record.fence_ref)?,
        stages: record.stages,
    })
}

/// Resolve an event record.
pub fn event(record: &EventRecord, resolver: &impl RefResolver) -> Result<EventOp, ResolveRefusal> {
    Ok(EventOp {
        kind: record.kind,
        event: one(resolver, record.event_ref)?,
        value: record.value,
    })
}

/// Resolve a barrier record, appending any list it names to the arena.
pub fn barrier(
    record: &BarrierRecord<'_>,
    resolver: &impl RefResolver,
    arenas: &mut ExecArenas,
) -> Result<BarrierOp, ResolveRefusal> {
    Ok(match *record {
        BarrierRecord::Resources {
            refs,
            after_stages,
            before_stages,
        } => BarrierOp {
            target: BarrierTarget::Resources(append_refs(arenas, resolver, refs)?),
            after_stages,
            before_stages,
        },
        BarrierRecord::Scope {
            scope,
            after_stages,
            before_stages,
            ..
        } => BarrierOp {
            target: BarrierTarget::Scope(scope),
            after_stages,
            before_stages,
        },
        BarrierRecord::Texture => BarrierOp {
            target: BarrierTarget::Texture,
            after_stages: None,
            before_stages: None,
        },
    })
}

/// Resolve any ordering record.
pub fn sync(
    record: &SyncRecord<'_>,
    resolver: &impl RefResolver,
    arenas: &mut ExecArenas,
) -> Result<SyncResolved, ResolveRefusal> {
    Ok(match record {
        SyncRecord::Fence(r) => SyncResolved::Fence(fence(r, resolver)?),
        SyncRecord::Event(r) => SyncResolved::Event(event(r, resolver)?),
        SyncRecord::Barrier(r) => SyncResolved::Barrier(barrier(r, resolver, arenas)?),
    })
}

/// The three ordering operations, which are three classes rather than one.
///
/// Returned as a small enum rather than folded into one class: a fence, an
/// event and a barrier order different things over different scopes, and the
/// census counts them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncResolved {
    Fence(FenceOp),
    Event(EventOp),
    Barrier(BarrierOp),
}

/// Resolve a residency declaration into the participations it asserts.
///
/// The window is the declared list; the usage and stages ride on the record.
/// Heaps and resources both land in the same arena because both are resolved
/// objects — what they *mean* is the record's `subject`, which the caller keeps.
pub fn residency(
    record: &ResidencyRecord<'_>,
    resolver: &impl RefResolver,
    arenas: &mut ExecArenas,
) -> Result<ResourceSpan, ResolveRefusal> {
    append_refs(arenas, resolver, record.refs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ObjectListRef, SlotGeneration};
    use reims_vgpu_protocol::closure::Rail;
    use reims_vgpu_protocol::residency::RenderStages;
    use reims_vgpu_protocol::sync::{BarrierScope, EventKind, FenceKind};

    /// A resolver over a fixed set of live refs.
    struct Live(Vec<u32>);

    impl RefResolver for Live {
        fn resource(&self, object_ref: u32) -> Option<ResourceId> {
            self.0.contains(&object_ref).then_some(ResourceId {
                slot: ObjectListRef(object_ref),
                generation: SlotGeneration(7),
            })
        }
    }

    fn id(object_ref: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(object_ref),
            generation: SlotGeneration(7),
        }
    }

    /// A copy's two endpoints resolve independently, and the operation carries
    /// ids rather than the numbers the guest wrote.
    #[test]
    fn a_transfer_resolves_both_of_its_endpoints() {
        let live = Live(vec![5151, 5252]);
        let record = BlitRecord::BufferToBuffer {
            source_ref: 5151,
            source_offset: 0x10,
            dest_ref: 5252,
            dest_offset: 0x20,
            size: 0x30,
        };
        assert_eq!(
            blit(&record, &live),
            Ok(BlitOp::BufferToBuffer {
                source: id(5151),
                source_offset: 0x10,
                dest: id(5252),
                dest_offset: 0x20,
                size: 0x30,
            })
        );
    }

    /// A ref naming nothing refuses with the guest's own number. It does not
    /// become a null id: an operation naming an object that does not exist
    /// would still be ordered against by everything downstream.
    #[test]
    fn a_dead_ref_refuses_by_name_rather_than_resolving_to_nothing() {
        let live = Live(vec![5151]);
        let record = BlitRecord::BufferToBuffer {
            source_ref: 5151,
            source_offset: 0,
            dest_ref: 4242,
            dest_offset: 0,
            size: 1,
        };
        assert_eq!(
            blit(&record, &live),
            Err(ResolveRefusal::UnknownRef { object_ref: 4242 })
        );
    }

    /// A whole-resource content record resolves to no subresource, and the
    /// `slice:level:` form to exactly one. `None` is not level zero.
    #[test]
    fn the_two_content_targets_stay_apart_through_resolution() {
        let live = Live(vec![7]);
        let whole = ResourceStateRecord {
            directive: reims_vgpu_protocol::resource_state::ContentDirective::Synchronize,
            target: RecordTarget::WholeResource { object_ref: 7 },
        };
        let sliced = ResourceStateRecord {
            directive: reims_vgpu_protocol::resource_state::ContentDirective::Synchronize,
            target: RecordTarget::SliceLevel {
                texture_ref: 7,
                slice: 0,
                level: 0,
            },
        };
        let whole = resource_state(&whole, &live).expect("resolved");
        let sliced = resource_state(&sliced, &live).expect("resolved");
        assert_eq!(
            whole.target,
            ResourceStateTarget::Resource {
                resource: id(7),
                subresource: None
            }
        );
        assert_ne!(whole.target, sliced.target);
    }

    /// A barrier's list lands in the arena and the operation names a window of
    /// it. The operation is two `u32` whatever the guest sent.
    #[test]
    fn a_barrier_list_goes_to_the_arena_and_the_operation_names_a_window() {
        let live = Live(vec![1, 2, 3]);
        let mut arenas = ExecArenas::default();
        let refs = [1u32, 2, 3].map(|v| RefBind {
            object_ref: reims_vgpu_protocol::decode::U32le::new(v),
        });
        let record = BarrierRecord::Resources {
            refs: &refs,
            after_stages: Some(RenderStages(1)),
            before_stages: Some(RenderStages(2)),
        };
        let op = barrier(&record, &live, &mut arenas).expect("resolved");
        let BarrierTarget::Resources(span) = op.target else {
            panic!("not a resource barrier");
        };
        assert_eq!(span, ResourceSpan { start: 0, len: 3 });
        assert_eq!(&arenas.resources[span.range()], &[id(1), id(2), id(3)]);

        // A second list starts where the first ended, so one arena serves the
        // whole packet.
        let second = barrier(&record, &live, &mut arenas).expect("resolved");
        let BarrierTarget::Resources(span) = second.target else {
            panic!("not a resource barrier");
        };
        assert_eq!(span, ResourceSpan { start: 3, len: 3 });
    }

    /// A list with a dead ref in it leaves nothing behind. Half a list in the
    /// arena behind a window nobody names is harmless right up until the arena
    /// is reused.
    #[test]
    fn a_refused_list_leaves_the_arena_as_it_found_it() {
        let live = Live(vec![1, 3]);
        let mut arenas = ExecArenas::default();
        let refs = [1u32, 2, 3].map(|v| RefBind {
            object_ref: reims_vgpu_protocol::decode::U32le::new(v),
        });
        let record = BarrierRecord::Resources {
            refs: &refs,
            after_stages: None,
            before_stages: None,
        };
        assert_eq!(
            barrier(&record, &live, &mut arenas),
            Err(ResolveRefusal::UnknownRef { object_ref: 2 })
        );
        assert!(arenas.resources.is_empty());
    }

    /// A scope barrier names no list, so it appends nothing.
    #[test]
    fn a_scope_barrier_touches_no_arena() {
        let live = Live(Vec::new());
        let mut arenas = ExecArenas::default();
        let record = BarrierRecord::Scope {
            scope: BarrierScope(BarrierScope::BUFFERS),
            after_stages: None,
            before_stages: None,
            unidentified_u8: None,
        };
        let op = barrier(&record, &live, &mut arenas).expect("resolved");
        assert_eq!(
            op.target,
            BarrierTarget::Scope(BarrierScope(BarrierScope::BUFFERS))
        );
        assert!(arenas.resources.is_empty());
    }

    /// The three ordering records reach three different classes.
    #[test]
    fn the_three_ordering_records_stay_three_classes() {
        let live = Live(vec![9]);
        let mut arenas = ExecArenas::default();
        let fence_record = SyncRecord::Fence(FenceRecord {
            kind: FenceKind::Update,
            fence_ref: 9,
            stages: None,
        });
        let event_record = SyncRecord::Event(EventRecord {
            kind: EventKind::Signal,
            event_ref: 9,
            value: 5,
        });
        let barrier_record = SyncRecord::Barrier(BarrierRecord::Texture);
        assert!(matches!(
            sync(&fence_record, &live, &mut arenas),
            Ok(SyncResolved::Fence(_))
        ));
        assert!(matches!(
            sync(&event_record, &live, &mut arenas),
            Ok(SyncResolved::Event(_))
        ));
        assert!(matches!(
            sync(&barrier_record, &live, &mut arenas),
            Ok(SyncResolved::Barrier(_))
        ));
        // The rail is not consulted here: a record has already been recognised
        // on its own rail by the time it reaches resolution.
        let _ = Rail::Render;
    }

    /// Every refusal reason is distinct, and a decode refusal keeps its own.
    #[test]
    fn refusal_reasons_are_distinct_and_a_decode_refusal_keeps_its_own() {
        let decode = DecodeRefusal::UnknownOpcode {
            rail: Rail::Blit,
            opcode: 1,
        };
        assert_eq!(
            ResolveRefusal::from(decode).reason(),
            decode.reason(),
            "a decode refusal must not be renamed on its way through"
        );
        let mine = [
            ResolveRefusal::UnknownRef { object_ref: 1 },
            ResolveRefusal::ArenaOverflow { wanted: 2 },
        ];
        let mut seen: Vec<&str> = mine.iter().map(|r| r.reason()).collect();
        seen.push(ResolveRefusal::from(decode).reason());
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(seen.len(), before);
    }
}
