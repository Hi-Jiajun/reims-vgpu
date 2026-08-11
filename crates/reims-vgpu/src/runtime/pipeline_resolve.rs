//! Resolving a draw's pipeline and both its shaders, once per pipeline object
//! rather than once per draw.
//!
//! # What this replaces, and what it cost
//!
//! Every draw chain used to walk the whole path from `pipeline_ref` to two
//! translated SPIR-V modules: an object-list read and a descriptor read for the
//! pipeline, a full TLV decode of that descriptor, then for each of the two
//! functions another object-list read, another descriptor read, a decode, and a
//! read of the whole MTLB container out of guest memory — followed by a linear
//! scan of each container for the wrapper magic and a content hash of each AIR
//! blob to key the translate cache.
//!
//! That is **eight guest page-table walks and five allocations per draw**, and
//! the answer is the same one every time: on a driven macos-13
//! sustained-animation boot, `pipeline_misses` and `shader_misses` are both
//! **zero** across 27 000 chains a second. `chain_phase`'s split of the span
//! (see [`crate::runtime::chain_phase::Phase::PipelineMtlb`]) priced the four
//! parts, per chain of 26.5 us:
//!
//! ```text
//! pl_desc_us    2.03 us   load_render_pipeline: 2 walks, 1 alloc, TLV decode
//! pl_mtlb_us    1.26 us   both load_mtlb: 6 walks, 4 allocs, 2 decodes
//! pl_xlate_us   1.19 us   both content hashes and the translate cache mutex
//! pl_air_us     0.20 us   both wrapper-magic scans
//! ```
//!
//! 4.68 us, **17.7 % of a draw chain**, spent arriving somewhere this device had
//! already been. A hit that costs what a compile costs is the thing this module
//! deletes.
//!
//! # The identity a memo entry is checked against
//!
//! A guest object's identity is its **object-list entry**, and this module's
//! whole correctness argument is that sentence. A 12-byte entry is the guest's
//! own authoritative statement of what a ref means — its type tag, its
//! descriptor's address and its descriptor's length — and the guest writes it
//! into shared memory with no doorbell, which is exactly why every rail here
//! re-reads it instead of caching what it once said.
//!
//! So this memo re-reads it too. [`resolve`] reads the three entries a draw
//! depends on — the pipeline object and both function objects — and serves the
//! cached resolution only when all three are byte-identical to the ones the
//! entry was built from. Three 12-byte reads is three page-table walks, ~0.6 us,
//! against the 4.68 us of work they authorise skipping.
//!
//! ## What that check does not cover, stated exactly
//!
//! An entry that has not changed permits two things this memo will not notice:
//!
//! - the guest **rewriting a descriptor in place**, at the same address and the
//!   same length, to mean a different object;
//! - the guest **rewriting the MTLB bytes in place**, at the `blob_gva` and
//!   `blob_size` the unchanged descriptor names, to hold a different shader.
//!
//! Neither is a shape Metal produces. A `MTLRenderPipelineState` and a
//! `MTLFunction` are immutable once created; a recompile is a new object, and a
//! new object gets a new descriptor allocation and therefore a new entry. But
//! "Metal does not do this" is a claim about a guest and not about the contract,
//! which is why it is written here rather than assumed, and why the memo is
//! switchable: `REIMS_VGPU_PIPELINE_MEMO=off` takes every chain back down the
//! full path, so a guest that ever contradicts the paragraph above can be
//! confirmed against a binary that cannot be wrong about it.
//!
//! The hazard this is **not** is page recycling. A memo entry holds an
//! `Arc<CachedShader>` and an `Arc<RenderPipelineDescriptor>` — host-side owned
//! copies — so it never reads through a stale guest pointer and holds no
//! reference over a guest page. The failure mode of a wrong entry is a stale
//! *shader*, which is a visual defect, not a memory-safety one.
//!
//! # Counters
//!
//! On `store_routes`, so a boot says which path it took rather than leaving it
//! inferred from a frame rate:
//!
//! | route | meaning |
//! |---|---|
//! | `pipe_memo_hit` | all three entries matched; the resolution was reused |
//! | `pipe_memo_miss` | no entry for this `(task, pipeline_ref)` |
//! | `pipe_memo_stale` | an entry existed and one of the three had changed |
//! | `pipe_memo_evict` | [`MEMO_CAPACITY`] pushed a resolution out |
//! | `pipe_memo_forget_all` | a device reset invalidated every key at once |
//! | `pipe_memo_off` | the memo is switched off; one per chain |
//!
//! `pipe_memo_stale` is the one to read. It is the population the paragraph
//! above says should be near zero on a steady desktop and non-zero only when a
//! guest genuinely replaces a pipeline; a boot where it tracks the hit count is
//! a boot where this memo is buying nothing and the check should be reconsidered
//! rather than the cap raised.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::hash::Hash;
use std::sync::{Arc, Mutex, OnceLock};

use crate::backend::vulkan::engine::DrawPreparationDecline;
use crate::model::DeviceState;
use crate::runtime::decode::resource::{ListObjectEntry, RenderPipelineDescriptor};
use crate::runtime::drain::note_store_route;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::m2v_cache::CachedShader;
use crate::runtime::mtlb::{load_mtlb, AirLoadRail};
use crate::runtime::objects;

/// How many `(task_id, pipeline_ref)` resolutions are held at once.
///
/// Sized against what a guest asks for rather than picked: a macOS desktop
/// compositing session drives a few dozen distinct pipeline objects per task
/// across a handful of live tasks, so this is roughly an order of magnitude of
/// headroom. `pipe_memo_evict` is the reading that says whether that is true on
/// a rail — a non-zero count means the working set exceeded this and the cap is
/// costing hits, which is the only thing that would justify raising it.
///
/// An entry is three `Arc` clones and a `[ListObjectEntry; 3]`; the shaders it
/// names are the same allocations the translate cache already holds, so the cap
/// bounds pointers rather than shader bytes.
pub const MEMO_CAPACITY: usize = 1024;

/// Everything a draw chain needs from its pipeline ref, resolved once.
///
/// Every field is an `Arc` because the whole point is that a hit copies nothing:
/// `RenderPipelineDescriptor` owns two `Vec`s (its vertex attributes and its
/// colour attachments) and cloning it per draw would put back a fraction of the
/// allocation traffic this module exists to remove.
#[derive(Clone)]
pub struct ResolvedRenderPipeline {
    pub desc: Arc<RenderPipelineDescriptor>,
    pub vertex: Arc<CachedShader>,
    pub fragment: Arc<CachedShader>,
}

/// The three object-list entries a resolution depends on, in the order they are
/// read: the pipeline object, then the vertex and fragment function objects.
///
/// A fixed-size array rather than three named fields because the only operation
/// on it is equality against a freshly-read one, and a named-field struct
/// invites a comparison that forgets a member. See the module doc for why the
/// entry is the identity.
type EntryTriple = [ListObjectEntry; 3];

struct Entry {
    identity: EntryTriple,
    resolved: ResolvedRenderPipeline,
}

/// A map that holds at most `CAP` entries, dropping the oldest *insertion* to
/// stay there.
///
/// The capacity is a const parameter rather than a field so it cannot be passed
/// wrong at a second construction site, and the map is private with `insert` as
/// its only mutator so a caller cannot reach `entries` and grow it past the
/// bound — `AGENTS.md`'s "make the invariant unrepresentable" rather than a scan
/// looking for places that forgot to check.
///
/// Oldest-insertion and not least-recently-used: a resolution's value does not
/// decay with time, and the population this bounds is pipeline objects a guest
/// creates at app launch and then keeps. LRU would buy a different eviction
/// order for a working set that does not exceed the cap at all —
/// `pipe_memo_evict` is what says whether that assumption holds on a rail.
struct BoundedByInsertion<K: Copy + Eq + Hash, V, const CAP: usize> {
    entries: HashMap<K, V>,
    order: VecDeque<K>,
}

impl<K: Copy + Eq + Hash, V, const CAP: usize> BoundedByInsertion<K, V, CAP> {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&self, key: &K) -> Option<&V> {
        self.entries.get(key)
    }

    /// File `value` under `key`, returning whatever the cap pushed out.
    ///
    /// Re-filing a live key replaces its value and does **not** queue a second
    /// slot in the order — otherwise the deque outgrows the map and starts
    /// evicting keys that are the newest thing in it.
    fn insert(&mut self, key: K, value: V) -> Option<K> {
        if self.entries.insert(key, value).is_some() {
            return None;
        }
        self.order.push_back(key);
        if self.order.len() <= CAP {
            return None;
        }
        let old = self.order.pop_front()?;
        self.entries.remove(&old);
        Some(old)
    }
}

type Memo = BoundedByInsertion<(u32, u32), Entry, MEMO_CAPACITY>;

fn memo() -> &'static Mutex<Memo> {
    static MEMO: OnceLock<Mutex<Memo>> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(Memo::new()))
}

/// Drop every resolution. Called from `device_reset`, which is where the keys
/// stop meaning anything — see the comment at that call site.
pub fn forget_all() {
    let mut m = memo().lock().unwrap_or_else(|e| e.into_inner());
    *m = Memo::new();
    note_store_route("pipe_memo_forget_all");
}

/// Whether the memo is on. See [`crate::env::PIPELINE_MEMO`].
fn memo_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            crate::env::read(crate::env::PIPELINE_MEMO).0,
            crate::env::Switch::Off
        )
    })
}

/// Read the object-list entries for a pipeline and the two functions it names.
///
/// `None` from any of the three is "the guest has not told us", which the full
/// path reports with its own rung — so a miss here does not refuse, it declines
/// to *serve from the memo* and lets [`resolve_uncached`] produce the named
/// failure. There is exactly one place a draw's pipeline failure is described
/// and it is not this function.
///
/// Neither func ref can be zero here. `ref == 0` is "no function bound" and
/// would read object-list slot 0 rather than refusing, but every triple this is
/// called with comes from a descriptor [`resolve_uncached`] already accepted,
/// and `load_render_pipeline` refuses a zero in either stage before returning
/// one.
fn read_identity<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
    vertex_ref: u32,
    fragment_ref: u32,
) -> Option<EntryTriple> {
    Some([
        objects::lookup_list_entry(state, host, task_id, pipeline_ref)?,
        objects::lookup_list_entry(state, host, task_id, vertex_ref)?,
        objects::lookup_list_entry(state, host, task_id, fragment_ref)?,
    ])
}

/// Resolve `pipeline_ref` to its descriptor and both translated shaders.
///
/// Serves a memoized resolution when the three object-list entries it was built
/// from still read identically; see the module doc for what that check is and
/// what it does not cover.
pub fn resolve<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> Result<ResolvedRenderPipeline, DrawPreparationDecline> {
    if !memo_enabled() {
        note_store_route("pipe_memo_off");
        return resolve_uncached(state, host, task_id, pipeline_ref);
    }

    // The func refs come from the cached entry rather than from a fresh
    // descriptor read: reading the descriptor to learn which functions to check
    // would pay most of what the memo is here to skip. The pipeline object's own
    // entry is the first of the three compared, so a pipeline that has been
    // replaced fails the check before its stale func refs are believed for
    // anything but the two reads that then also fail it.
    let cached = {
        let m = memo().lock().unwrap_or_else(|e| e.into_inner());
        m.get(&(task_id, pipeline_ref))
            .map(|e| (e.identity, e.resolved.clone()))
    };
    if let Some((identity, resolved)) = cached {
        let fresh = read_identity(
            state,
            host,
            task_id,
            pipeline_ref,
            resolved.desc.vertex_func_ref,
            resolved.desc.fragment_func_ref,
        );
        if fresh == Some(identity) {
            note_store_route("pipe_memo_hit");
            return Ok(resolved);
        }
        note_store_route("pipe_memo_stale");
    } else {
        note_store_route("pipe_memo_miss");
    }

    let resolved = resolve_uncached(state, host, task_id, pipeline_ref)?;
    // Built after the resolution and from the same refs it used, so an entry can
    // only be filed under an identity that was readable at the moment the
    // resolution was taken. An unreadable one files nothing rather than filing a
    // resolution no later read can invalidate.
    if let Some(identity) = read_identity(
        state,
        host,
        task_id,
        pipeline_ref,
        resolved.desc.vertex_func_ref,
        resolved.desc.fragment_func_ref,
    ) {
        let evicted = memo().lock().unwrap_or_else(|e| e.into_inner()).insert(
            (task_id, pipeline_ref),
            Entry {
                identity,
                resolved: resolved.clone(),
            },
        );
        if evicted.is_some() {
            note_store_route("pipe_memo_evict");
        }
    }
    Ok(resolved)
}

/// The full path: object list → descriptor → decode → MTLB → AIR → SPIR-V, for
/// the pipeline and both of its functions.
///
/// This is the only place a draw's pipeline resolution can fail, and each of its
/// seven refusals keeps the `DrawPreparationDecline` variant it always had — the
/// memo in front of it neither adds a failure nor renames one.
fn resolve_uncached<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> Result<ResolvedRenderPipeline, DrawPreparationDecline> {
    let desc = crate::runtime::draw::load_render_pipeline(state, host, task_id, pipeline_ref)
        .ok_or(DrawPreparationDecline::PipelineMissing {
            task_id,
            pipeline_ref,
        })?;
    // The same three sub-phases the call site used to open around this work,
    // moved in with it. They are inert outside a live `ChainTimer`, so the two
    // non-draw callers of the loaders below are unaffected — and on the draw
    // rail `pl_desc_us` now brackets the memo's own identity check, which is
    // what makes the hit path's cost readable against the miss path's.
    use crate::runtime::chain_phase::{enter, Phase};
    enter(Phase::PipelineMtlb);
    let v_mtlb = load_mtlb(
        state,
        host,
        task_id,
        desc.vertex_func_ref,
        AirLoadRail::Draw,
    )
    .ok_or(DrawPreparationDecline::VertexMtlbMissing {
        task_id,
        function_ref: desc.vertex_func_ref,
    })?;
    let f_mtlb = load_mtlb(
        state,
        host,
        task_id,
        desc.fragment_func_ref,
        AirLoadRail::Draw,
    )
    .ok_or(DrawPreparationDecline::FragmentMtlbMissing {
        task_id,
        function_ref: desc.fragment_func_ref,
    })?;
    enter(Phase::PipelineAir);
    let v_air = crate::runtime::mtlb::extract_air(&v_mtlb).map_err(|reason| {
        DrawPreparationDecline::VertexAirExtract {
            function_ref: desc.vertex_func_ref,
            reason,
        }
    })?;
    let f_air = crate::runtime::mtlb::extract_air(&f_mtlb).map_err(|reason| {
        DrawPreparationDecline::FragmentAirExtract {
            function_ref: desc.fragment_func_ref,
            reason,
        }
    })?;
    enter(Phase::PipelineXlate);
    let vertex = crate::runtime::m2v_cache::translate_cached_reflected(
        v_air,
        metal2vulkan::passes::Stage::Vertex,
        pipeline_ref,
    )
    .map_err(|reason| DrawPreparationDecline::VertexTranslate {
        pipeline_ref,
        reason,
    })?;
    let fragment = crate::runtime::m2v_cache::translate_cached_reflected(
        f_air,
        metal2vulkan::passes::Stage::Fragment,
        pipeline_ref,
    )
    .map_err(|reason| DrawPreparationDecline::FragmentTranslate {
        pipeline_ref,
        reason,
    })?;
    Ok(ResolvedRenderPipeline {
        desc: Arc::new(desc),
        vertex,
        fragment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(gva: u64, len: u32, ot: u8) -> ListObjectEntry {
        ListObjectEntry {
            object_type: ot,
            descriptor_length: len,
            descriptor_gva: gva,
        }
    }

    /// The cap has to evict, or a guest that cycles pipeline refs grows this map
    /// without bound for the life of the VM.
    #[test]
    fn the_capacity_evicts_the_oldest_insertion() {
        let mut m: BoundedByInsertion<u32, u32, 4> = BoundedByInsertion::new();
        assert_eq!(m.insert(1, 10), None, "under the cap evicts nothing");
        for k in 2..=4 {
            assert_eq!(m.insert(k, k * 10), None);
        }
        assert_eq!(m.insert(5, 50), Some(1), "the oldest insertion is named");
        assert_eq!(m.entries.len(), 4, "the cap holds");
        assert_eq!(m.get(&1), None, "and it is gone");
        assert_eq!(m.get(&5), Some(&50));
    }

    /// Re-inserting a live key must not queue a second eviction slot for it, or
    /// the order deque outgrows the map and evicts entries that are still the
    /// newest thing in it.
    #[test]
    fn re_inserting_a_key_does_not_grow_the_order() {
        let mut m: BoundedByInsertion<u32, u32, 4> = BoundedByInsertion::new();
        for v in 0..64 {
            assert_eq!(m.insert(5, v), None, "a replacement evicts nothing");
        }
        assert_eq!(m.entries.len(), 1);
        assert_eq!(m.order.len(), 1, "one key, one slot in the order");
        assert_eq!(m.get(&5), Some(&63), "and the newest value won");
    }

    /// The identity is compared as a whole. A change to any of the three
    /// entries, in any of their three fields, has to read as different — this is
    /// the check the module's whole correctness argument rests on.
    #[test]
    fn every_field_of_every_entry_is_part_of_the_identity() {
        let base: EntryTriple = [
            entry(0x1000, 64, 7),
            entry(0x2000, 32, 6),
            entry(0x3000, 32, 6),
        ];
        for slot in 0..3 {
            let mut gva = base;
            gva[slot].descriptor_gva += 1;
            assert_ne!(base, gva, "slot {slot} descriptor_gva");
            let mut len = base;
            len[slot].descriptor_length += 1;
            assert_ne!(base, len, "slot {slot} descriptor_length");
            let mut ot = base;
            ot[slot].object_type += 1;
            assert_ne!(base, ot, "slot {slot} object_type");
        }
    }
}
