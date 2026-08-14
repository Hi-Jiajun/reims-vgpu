//! Resource-validity ownership for render targets.
//!
//! A render Store preserves pixels in the host attachment. It does not imply a
//! host-to-guest transfer. The guest makes that transfer observable by naming
//! the resource in `CmdSynchronizeResources`, or this device needs the guest
//! bytes itself for a fallback reader. Until then, [`PendingWritebacks`] records
//! that the engine image is authoritative and repeated Stores into the resource
//! replace one another without touching guest RAM.
//!
//! # A resource owns its transfer backing
//!
//! Type-11 debts carry a mapping id, geometry, and map generation. GVA debts
//! carry the task-local texture reference, GVA declaration, geometry, format,
//! and resource generation. The live GVA resource separately retains the
//! ordered physical pages of its transfer backing. Ordinary task unmap changes
//! virtual-address bookkeeping but does not retarget that resource. Explicit
//! discard drops the transfer backing, and the next prepare or synchronize
//! resolves it again without replacing the host texture.
//!
//! This is the safety property the former deferred-window design lacked: it
//! parked raw host pointers across guest execution. This model retains page
//! identities, not pointers; every transfer still constructs bounded
//! `GuestSlice`s from the owning RAMBlock import.
//!
//! # Validity transitions decide direction
//!
//! A GPU Store makes the host image authoritative. A later guest
//! `clear_host_valid` makes the guest copy newer; payment then abandons the host
//! image rather than overwriting the guest's work. Surface resources use
//! `ResourceValidity`'s ordered sequence. Task-GVA resources use the validity
//! generation keyed by `(task, texture_ref)`, including the case where that
//! integer collides with an unrelated mapping id.
//!
//! A named synchronize pays only its object list through
//! [`submit_for_resources`]. Readers that know a mapping or texture call
//! [`pay_for_mapping`] or [`pay_for_texture`]. Only a genuinely unnameable
//! aliasing reader uses [`pay_all`]. Completion stamps alone do not publish
//! resources.
//!
//! The engine's `gpu_only_content` flag keeps an unpaid image alive. A
//! successful payment calls `note_resident_content_copied_out`; replacement,
//! invalidation, task retirement, and generation movement release the same
//! ownership without inventing a guest write.
//!
//! [`MAX_DEBTS`] bounds only anonymous type-11 surface debts. GVA resource
//! lifetime is explicit — resource discard/delete and task teardown — so an
//! unrelated capacity limit must not invent an early synchronization point.

use crate::model::DeviceState;
use crate::runtime::host::{HostMemory, HostOps};

/// Debts held at once, before an arm pays the oldest to make room.
///
/// This is the existing measured ceiling for the ledger, now shared by both
/// backing representations rather than duplicated per representation. An
/// insertion past it pays the oldest frame, so the bound can cost coalescing but
/// cannot lose pixels. `wbdebt_evicted` reports when a workload reaches it.
pub const MAX_DEBTS: usize = 32;

/// A frame owed to one type-11 mapping's guest pages.
///
/// Deliberately four integers and no memory. See the module doc: the rail this
/// replaces held resolved host pointers and corrupted the guest's page tables
/// with them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WritebackDebt {
    /// Geometry the Store was taken at, and the geometry the payment writes.
    pub width: u32,
    pub height: u32,
    /// `MappingEntry::map_generation` at the arm.
    ///
    /// The identity the payment re-derives carries this, so a mapping the guest
    /// has since remapped produces a different identity — a different resident —
    /// and the debt is void rather than paid into the wrong pages.
    pub map_generation: u32,
    /// Arm order, for choosing which debt an over-full ledger pays first.
    pub seq: u64,
}

/// The guest resource that owns one GVA render attachment.
///
/// Unlike the address, this is also what `CmdSynchronizeResources` names. A
/// task is part of the key because object references are task-local.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GvaResourceKey {
    pub task_id: u32,
    pub texture_ref: u32,
}

/// A frame held only by a GVA target's engine-resident image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GvaWritebackDebt {
    pub gva: u64,
    pub row_stride: u32,
    pub width: u32,
    pub height: u32,
    pub format: u16,
    pub generation: u64,
    pub guest_write: crate::runtime::buffer_write_gen::BufferWriteStamp,
    pub seq: u64,
}

/// The transfer backing retained by one live GVA texture resource.
///
/// The resource owns this physical-page identity after its virtual declaration
/// has been resolved. Task unmap changes the task's CPU mapping bookkeeping; it
/// does not retarget a live resource. An explicit resource discard drops only
/// `pages`, allowing the next prepare/synchronize to establish a new transfer
/// backing without changing the host texture's identity.
#[derive(Clone, Debug)]
struct GvaResourceState {
    generation: u64,
    gva: u64,
    span: u64,
    pages: Option<std::sync::Arc<[u64]>>,
}

/// One entry in the bounded ledger, irrespective of backing kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum WritebackKey {
    Mapping(u32),
}

/// Every render resource whose current frame exists only in a host resident.
///
/// Surface resources key by mapping id; GVA resources key by their task-local
/// texture reference. In either representation, a second Store replaces the
/// first rather than queueing another frame.
#[derive(Debug, Default)]
pub struct PendingWritebacks {
    debts: std::collections::BTreeMap<u32, WritebackDebt>,
    gva_debts: std::collections::BTreeMap<GvaResourceKey, GvaWritebackDebt>,
    gva_resources: std::collections::BTreeMap<GvaResourceKey, GvaResourceState>,
    next_seq: u64,
    next_gva_generation: u64,
}

impl PendingWritebacks {
    /// Mappings currently owed a frame.
    pub fn len(&self) -> usize {
        self.debts.len() + self.gva_debts.len()
    }

    /// Whether anything is owed at all — the check every reader makes, and the
    /// one that has to be free.
    pub fn is_empty(&self) -> bool {
        self.debts.is_empty() && self.gva_debts.is_empty()
    }

    /// What `mapping_id` is owed, if anything.
    pub fn get(&self, mapping_id: u32) -> Option<WritebackDebt> {
        self.debts.get(&mapping_id).copied()
    }

    /// Take `mapping_id`'s debt, leaving it owed nothing.
    pub fn take(&mut self, mapping_id: u32) -> Option<WritebackDebt> {
        self.debts.remove(&mapping_id)
    }

    /// Every mapping owed a frame, oldest arm first.
    pub fn mappings_by_age(&self) -> Vec<u32> {
        let mut all: Vec<(u64, u32)> = self.debts.iter().map(|(id, d)| (d.seq, *id)).collect();
        all.sort_unstable();
        all.into_iter().map(|(_, id)| id).collect()
    }

    /// The surface mapping whose debt has been owed longest.
    fn oldest(&self) -> Option<WritebackKey> {
        self.debts
            .iter()
            .min_by_key(|(_, d)| d.seq)
            .map(|(id, _)| WritebackKey::Mapping(*id))
    }

    /// Record that `mapping_id` is owed a frame, returning the mapping whose
    /// debt the caller must pay to bring the ledger back under [`MAX_DEBTS`] —
    /// `None` in the ordinary case.
    ///
    /// A mapping already owed a frame is *replaced*: the later frame is the
    /// fresher answer and the earlier one has been superseded on the GPU
    /// already. That replacement is the whole saving, so it is counted.
    ///
    /// The over-full entry is left in the ledger and handed back by name rather
    /// than removed here, so [`PendingWritebacks::take`] stays the only way a
    /// debt leaves — a removal that is not a payment is a frame the guest asked
    /// for and never received.
    #[must_use = "an evicted mapping still owes a frame and the caller must pay it"]
    pub fn arm(
        &mut self,
        mapping_id: u32,
        width: u32,
        height: u32,
        map_generation: u32,
    ) -> Option<WritebackKey> {
        let evict = match self.debts.len() >= MAX_DEBTS && !self.debts.contains_key(&mapping_id) {
            true => self.oldest(),
            false => None,
        };
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        let previous = self.debts.insert(
            mapping_id,
            WritebackDebt {
                width,
                height,
                map_generation,
                seq,
            },
        );
        if previous.is_some() {
            crate::runtime::drain::note_store_route("wbdebt_superseded");
        }
        crate::runtime::drain::note_store_route("wbdebt_armed");
        evict
    }

    /// Record a host-authoritative frame for one GVA resource.
    ///
    /// A second Store through the same task-local texture reference replaces
    /// the earlier debt. The returned previous debt names an older resident
    /// identity that the caller must release when the declaration changed.
    #[must_use = "a replaced resource debt may own an older resident identity"]
    pub fn arm_gva(
        &mut self,
        key: GvaResourceKey,
        mut debt: GvaWritebackDebt,
    ) -> Option<GvaWritebackDebt> {
        debt.seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        let previous = self.gva_debts.insert(key, debt);
        if previous.is_some() {
            crate::runtime::drain::note_store_route("gvadebt_superseded");
        }
        crate::runtime::drain::note_store_route("gvadebt_armed");
        previous
    }

    /// Establish or retrieve the lifetime identity of one live GVA resource.
    ///
    /// `pages` is accepted only on the first resolution after construction or
    /// explicit discard. Repeated draws and ordinary task unmaps keep the
    /// retained physical backing and the same host-texture generation.
    pub fn ensure_gva_resource(
        &mut self,
        key: GvaResourceKey,
        gva: u64,
        span: u64,
        pages: Option<Vec<u64>>,
    ) -> u64 {
        if let Some(resource) = self.gva_resources.get_mut(&key) {
            if resource.gva == gva && resource.span == span && resource.pages.is_none() {
                resource.pages = pages.map(std::sync::Arc::from);
            }
            return resource.generation;
        }
        self.next_gva_generation = self.next_gva_generation.wrapping_add(1);
        if self.next_gva_generation == 0 {
            self.next_gva_generation = 1;
        }
        let generation = self.next_gva_generation;
        self.gva_resources.insert(
            key,
            GvaResourceState {
                generation,
                gva,
                span,
                pages: pages.map(std::sync::Arc::from),
            },
        );
        generation
    }

    #[cfg(any(feature = "backend-vulkan", test))]
    fn gva_resource_backing(
        &self,
        key: GvaResourceKey,
    ) -> Option<(u64, u64, u64, std::sync::Arc<[u64]>)> {
        let resource = self.gva_resources.get(&key)?;
        Some((
            resource.generation,
            resource.gva,
            resource.span,
            std::sync::Arc::clone(resource.pages.as_ref()?),
        ))
    }

    #[cfg(any(feature = "backend-vulkan", test))]
    fn gva_resource_status(&self, key: GvaResourceKey) -> Option<(u64, u64, u64, bool)> {
        self.gva_resources.get(&key).map(|resource| {
            (
                resource.generation,
                resource.gva,
                resource.span,
                resource.pages.is_some(),
            )
        })
    }

    /// Release the transfer buffer of each named resource while preserving its
    /// host texture and lifetime identity.
    pub fn discard_gva_resources(&mut self, task_id: u32, object_ids: &[u32]) -> usize {
        let mut discarded = 0;
        for &texture_ref in object_ids {
            let key = GvaResourceKey {
                task_id,
                texture_ref,
            };
            if let Some(resource) = self.gva_resources.get_mut(&key) {
                discarded += usize::from(resource.pages.take().is_some());
            }
        }
        discarded
    }

    fn retire_gva_resource(&mut self, key: GvaResourceKey) -> (bool, Option<GvaWritebackDebt>) {
        let existed = self.gva_resources.remove(&key).is_some();
        (existed, self.gva_debts.remove(&key))
    }

    pub fn get_gva(&self, key: GvaResourceKey) -> Option<GvaWritebackDebt> {
        self.gva_debts.get(&key).copied()
    }

    pub fn has_gva(&self, key: GvaResourceKey) -> bool {
        self.gva_debts.contains_key(&key)
    }

    pub fn take_gva(&mut self, key: GvaResourceKey) -> Option<GvaWritebackDebt> {
        self.gva_debts.remove(&key)
    }

    /// Put back a debt whose guest backing was temporarily unavailable.
    /// Preserves its original age: inability to pay does not make an old frame
    /// the newest member of the ledger.
    #[cfg(feature = "backend-vulkan")]
    fn restore_gva(&mut self, key: GvaResourceKey, debt: GvaWritebackDebt) {
        let previous = self.gva_debts.insert(key, debt);
        debug_assert!(
            previous.is_none(),
            "a taken debt restores into its own hole"
        );
    }

    fn gvas_by_age(&self) -> Vec<GvaResourceKey> {
        let mut all: Vec<(u64, GvaResourceKey)> = self
            .gva_debts
            .iter()
            .map(|(key, debt)| (debt.seq, *key))
            .collect();
        all.sort_unstable();
        all.into_iter().map(|(_, key)| key).collect()
    }

    fn gvas_for_task(&self, task_id: u32) -> Vec<GvaResourceKey> {
        self.gva_resources
            .keys()
            .filter(|key| key.task_id == task_id)
            .copied()
            .collect()
    }

    #[cfg(feature = "backend-vulkan")]
    fn gva_for_identity(
        &self,
        identity: &crate::backend::vulkan::engine::TargetIdentity,
    ) -> Option<(GvaResourceKey, GvaWritebackDebt)> {
        let crate::backend::vulkan::engine::TargetIdentity::Gva {
            gva,
            width,
            height,
            generation,
            ..
        } = *identity
        else {
            return None;
        };
        self.gva_debts
            .iter()
            .find(|(_, debt)| {
                debt.gva == gva
                    && debt.width == width
                    && debt.height == height
                    && debt.generation == generation
            })
            .map(|(key, debt)| (*key, *debt))
    }
}

/// Whether the lazy rail is on for this process.
///
/// Read once. The rail changes *when* a frame reaches guest pages, not what the
/// bytes are, so a boot that flipped it midway would be two devices in one log.
pub fn lazy_writeback_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let (state, value) = crate::env::read(crate::env::LAZY_WRITEBACK);
        // Only an explicit `off` narrows to the eager Store. Unset, `on` and an
        // unrecognized value are all the shipping rail, which is what makes
        // `Switch::Unrecognized` — an operator's typo — fail toward the measured
        // default rather than silently selecting the arm it is 45 % slower on.
        let on = !matches!(state, crate::env::Switch::Off);
        crate::observe::off(format!(
            "lazy_writeback on={on} switch={state:?} value={}",
            value.unwrap_or_else(|| "<unset>".into())
        ));
        on
    })
}

/// Pay `mapping_id`'s owed frame, if it owes one.
///
/// The one call a reader of a named mapping's guest bytes makes before it reads
/// them. Free when nothing is owed — one `BTreeMap` emptiness check, which is
/// the answer on nearly every call.
pub fn pay_for_mapping<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
) {
    if state.pending_writebacks.is_empty() {
        return;
    }
    let Some(debt) = state.pending_writebacks.take(mapping_id) else {
        return;
    };
    pay(state, host, mapping_id, debt, "wbdebt_paid_named");
}

/// Pay every owed frame.
///
/// For a reader that cannot name the mapping it is about to read — a GVA span, a
/// buffer, a page walk that may alias a surface. Aliasing across the id
/// namespaces is real, so "cannot say" resolves to "owes all of them".
///
/// # Why the disjointness closures those readers already carry do not narrow it
///
/// The three GVA readers each build the exact page list they are about to touch,
/// and hand it to
/// [`crate::runtime::render_writeback::settle_guest_writes_unless_disjoint`] so a
/// reader somewhere else entirely does not wait for a surface's writeback. That
/// narrowing cannot be reused here, and the reason is the rail itself: the test
/// runs only when a copy is **outstanding**, and an owed frame has not been
/// submitted at all. With the lazy rail on, the common state is a clear debt
/// flag and a full ledger, where the closure never runs.
///
/// Narrowing this would need a page-extent hint held per debt, and a hint is the
/// beginning of holding resolved memory — which is what the module doc says this
/// rail must not do. [`note_unnamed_reach`] is the instrument that says whether
/// it would be worth it; read its doc before building one.
pub fn pay_all<M: HostMemory + HostOps>(state: &mut DeviceState, host: &mut M) {
    if state.pending_writebacks.is_empty() {
        return;
    }
    for mapping_id in state.pending_writebacks.mappings_by_age() {
        let Some(debt) = state.pending_writebacks.take(mapping_id) else {
            continue;
        };
        pay(state, host, mapping_id, debt, "wbdebt_paid_all");
    }
    for key in state.pending_writebacks.gvas_by_age() {
        let Some(debt) = state.pending_writebacks.take_gva(key) else {
            continue;
        };
        let _ = pay_gva(state, host, key, debt, GvaPaySite::All);
    }
}

/// One call in [`REACH_SAMPLE`] does the walk; the rest cost one modulo.
///
/// The walk is ~2 000 page-table descents for a 1080p span and the site that
/// dominates [`pay_all`] runs about 1 700 times a second, so measuring every call
/// would cost more than the rail saves and would be measuring the instrument.
/// A census wants a rate and not a total, and a rate converges on a 1-in-64
/// sample of a population this size: ~26 walks a second against ~1 700 calls.
const REACH_SAMPLE: u64 = 64;

/// Calls to [`note_unnamed_reach`], for the sample.
static REACH_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Does an unnameable reader that pays every debt actually read the pages it is
/// paying for?
///
/// # The question, and why it decides the rail's ceiling
///
/// The premise the lazy rail was built on read the settle census wrong. Those
/// counters count settles that **waited**, and on a driven macos-13
/// sustained-animation boot they total six a second — which is what "840 writes
/// consumed six times" was derived from. The *calls* are a different population:
/// `draw::vulkan::load_linear_guest_memoized` alone reaches its settle about
/// 1 700 times a second, reads the guest pages every one of them, and cannot name
/// a mapping, so it pays every owed frame. That is why the first driven on-arm
/// boot coalesced 130 Stores of 577 rather than the ~95 % the premise predicted.
///
/// But paying is only *owed* where the read and the surface share pages. A
/// compositor sampling a glyph atlas while three windows owe frames pays three
/// copies it will not look at. This counts which it is:
///
/// * `wbdebt_reach_overlap` — the sampled read touched a page some debt's
///   mapping holds. The payment was owed and no narrowing can remove it.
/// * `wbdebt_reach_disjoint` — it did not. The payment was pure waste, and the
///   ratio of these two is the prize a page-extent hint per debt would collect.
/// * `wbdebt_reach_unnamed` — the reader's own walk came back short, so nothing
///   could be ruled out. A narrowing must treat this as overlap.
///
/// `pages` is the reader's own closure, the same one it hands the disjointness
/// test, so both ends of the comparison stay one rule. It runs only on a sampled
/// call and only while something is owed.
pub fn note_unnamed_reach(state: &DeviceState, pages: impl FnOnce() -> Option<Vec<u64>>) {
    if state.pending_writebacks.is_empty() {
        return;
    }
    let n = REACH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if !n.is_multiple_of(REACH_SAMPLE) {
        return;
    }
    let Some(read) = pages() else {
        crate::runtime::drain::note_store_route("wbdebt_reach_unnamed");
        return;
    };
    let read: std::collections::BTreeSet<u64> = read.into_iter().collect();
    let overlap = state
        .pending_writebacks
        .mappings_by_age()
        .into_iter()
        .any(|mapping_id| {
            state
                .mapping_reach_pages(mapping_id)
                .is_some_and(|owed| owed.iter().any(|page| read.contains(page)))
        });
    match overlap {
        true => crate::runtime::drain::note_store_route("wbdebt_reach_overlap"),
        false => crate::runtime::drain::note_store_route("wbdebt_reach_disjoint"),
    }
}

/// Pay whatever a *texture* reference names, for a reader that reaches guest
/// bytes through a task GVA but knows which resource it is reading.
///
/// # Why a GVA reader is nameable after all, and what that measured
///
/// The three linear readers walk raw task GVAs, so the first cut of this rail
/// had them pay every owed frame. [`note_unnamed_reach`] priced that: **173
/// sampled payments over one driven macos-13 sustained-animation boot, 173
/// disjoint from every owed surface and not one overlap**, at a 1-in-64 sample
/// of ~11 000 payments. Meanwhile `wbdebt_paid_all` was 20 391 against
/// `wbdebt_paid_named` 755, so 96 % of all payments were the ones that read
/// nothing they paid for, and they cost `sampled_us` 1.64 → 8.49 us a chain.
///
/// They are nameable because the guest names them. A debt is keyed by mapping
/// id, and this device holds two ways from a texture reference to one:
/// `DeviceState::texture_to_mapping` for the per-task registration, and the id
/// itself where the guest uses one namespace for both —
/// [`crate::runtime::resource_validity::apply`] resolves a validity statement
/// through exactly this pair, and this is the same question asked of the same
/// two tables.
///
/// A reference that resolves to neither names no mapping this device holds, so
/// no debt can be about it. That is a statement about the registries and not
/// about a workload — but it is not a statement about raw *page* aliasing, where
/// a surface's pages are re-used as some other resource's backing with no
/// mapping entry. [`note_unnamed_reach`] stays wired at these sites as the
/// standing alarm for exactly that: it samples the read's own page walk against
/// every owed surface's pages, and `wbdebt_reach_overlap` above zero is a
/// payment this naming skipped and should not have.
pub fn pay_for_texture<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
) {
    if state.pending_writebacks.is_empty() {
        return;
    }
    let gva_key = GvaResourceKey {
        task_id,
        texture_ref,
    };
    let mut named = false;
    if let Some(debt) = state.pending_writebacks.take_gva(gva_key) {
        named = true;
        let _ = pay_gva(state, host, gva_key, debt, GvaPaySite::Named);
    }
    // Both surface spellings, in the order `resource_validity::apply` uses: a
    // reference that is itself a mapping id, and the per-task registration.
    // Paying one leaves the ledger holding the other, so asking twice costs a
    // map lookup and cannot pay the wrong surface.
    let mapped = state
        .texture_to_mapping
        .get(&(task_id, texture_ref))
        .copied();
    if state.pending_writebacks.get(texture_ref).is_some() {
        named = true;
        pay_for_mapping(state, host, texture_ref);
    }
    if let Some(mapping_id) = mapped.filter(|&id| id != texture_ref) {
        if state.pending_writebacks.get(mapping_id).is_some() {
            named = true;
            pay_for_mapping(state, host, mapping_id);
        }
    }
    if !named {
        crate::runtime::drain::note_store_route("wbdebt_texture_owes_nothing");
    }
}

/// The stable host-texture identity for one task-local GVA resource.
///
/// The first successful resolution retains the ordered physical pages that the
/// resource's transfer buffer names. Later calls return the same generation and
/// backing even if the task removes its virtual mapping. After explicit
/// discard, the next call may establish a replacement transfer backing while
/// preserving the host texture's generation.
#[cfg(feature = "backend-vulkan")]
pub fn gva_resource_generation<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    key: GvaResourceKey,
    gva: u64,
    span: u64,
) -> u64 {
    if let Some((generation, declared_gva, declared_span, has_pages)) =
        state.pending_writebacks.gva_resource_status(key)
    {
        if declared_gva != gva || declared_span != span {
            crate::observe::fail(format!(
                "gva_resource_refused task={} texture={} reason=declaration_changed",
                key.task_id, key.texture_ref
            ));
            return 0;
        }
        if has_pages {
            return generation;
        }
    }
    let page_size = state.page_size();
    let ordered = crate::runtime::gva_mem::task_gva_page_gpas(
        host,
        &state.tasks,
        key.task_id,
        gva,
        span,
        state.page_shift,
    );
    let want = reims_vgpu_paging::span::pages_spanned(gva, span, page_size);
    let pages = (ordered.len() as u64 == want).then_some(ordered);
    state
        .pending_writebacks
        .ensure_gva_resource(key, gva, span, pages)
}

/// Record a GVA render result as host-authoritative without touching guest
/// pages. Returns `false` when the attachment has no resource identity and must
/// use the eager transfer path.
#[cfg(feature = "backend-vulkan")]
pub fn arm_gva<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    _host: &mut M,
    task_id: u32,
    c0: &crate::runtime::draw::ColorRtRequest,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
) -> bool {
    let Some(generation) = (match *identity {
        crate::backend::vulkan::engine::TargetIdentity::Gva { generation, .. } => Some(generation),
        _ => None,
    }) else {
        return false;
    };
    if c0.texture_ref == 0 || generation == 0 {
        return false;
    }
    // Every older host-side spelling of this resource is stale as soon as the
    // render finishes. In particular, a compute storage resident and the
    // linear byte cache can otherwise sit above the guest-page reader and serve
    // the frame that preceded this Store indefinitely.
    state.invalidate_object_host_copies(task_id, c0.texture_ref);
    crate::runtime::surface_cache::evict_gva(state, c0.target_gva);
    let key = GvaResourceKey {
        task_id,
        texture_ref: c0.texture_ref,
    };
    let debt = GvaWritebackDebt {
        gva: c0.target_gva,
        row_stride: c0.row_stride,
        width: c0.width,
        height: c0.height,
        format: c0.format,
        generation,
        guest_write: state.buffer_write_gen.stamp(task_id, c0.texture_ref),
        seq: 0,
    };
    let previous = state.pending_writebacks.arm_gva(key, debt);
    if let Some(previous) = previous.filter(|previous| !same_gva_identity(*previous, debt)) {
        release_gva(previous);
    }
    true
}

/// Whether this exact GVA resident is the host-authoritative copy named by an
/// unpaid resource debt.
#[cfg(feature = "backend-vulkan")]
pub fn gva_resident_authoritative(
    state: &DeviceState,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
) -> bool {
    let Some((key, debt)) = state.pending_writebacks.gva_for_identity(identity) else {
        return false;
    };
    state
        .buffer_write_gen
        .stamp(key.task_id, key.texture_ref)
        .quiet_since(debt.guest_write)
}

/// Retire host-authoritative resources whose task-local references are about to
/// be replaced. The pixels are deliberately not copied: after this lifecycle
/// transition the old object no longer names guest storage to synchronize.
pub fn retire_gva_for_task(state: &mut DeviceState, task_id: u32) -> usize {
    let keys = state.pending_writebacks.gvas_for_task(task_id);
    let mut retired = 0;
    for key in keys {
        let (_, debt) = state.pending_writebacks.retire_gva_resource(key);
        retired += 1;
        #[cfg(feature = "backend-vulkan")]
        if let Some(debt) = debt {
            release_gva(debt);
        }
        #[cfg(not(feature = "backend-vulkan"))]
        let _ = debt;
    }
    if retired != 0 {
        crate::runtime::drain::note_store_route_n("gvadebt_retired_task", retired as u64);
    }
    retired
}

/// Retire one resource at its explicit lifetime boundary.
pub fn retire_gva_resource(state: &mut DeviceState, task_id: u32, texture_ref: u32) -> bool {
    let key = GvaResourceKey {
        task_id,
        texture_ref,
    };
    let (existed, debt) = state.pending_writebacks.retire_gva_resource(key);
    #[cfg(feature = "backend-vulkan")]
    if let Some(debt) = debt {
        release_gva(debt);
    }
    #[cfg(not(feature = "backend-vulkan"))]
    let _ = debt;
    existed || debt.is_some()
}

/// Release named resources' retained transfer backings.
pub fn discard_gva_resources(state: &mut DeviceState, task_id: u32, object_ids: &[u32]) -> usize {
    state
        .pending_writebacks
        .discard_gva_resources(task_id, object_ids)
}

#[cfg(feature = "backend-vulkan")]
fn same_gva_identity(a: GvaWritebackDebt, b: GvaWritebackDebt) -> bool {
    a.gva == b.gva
        && a.width == b.width
        && a.height == b.height
        && a.generation == b.generation
        && a.format == b.format
}

#[cfg(feature = "backend-vulkan")]
fn gva_identity(debt: GvaWritebackDebt) -> crate::backend::vulkan::engine::TargetIdentity {
    crate::backend::vulkan::engine::TargetIdentity::Gva {
        gva: debt.gva,
        width: debt.width,
        height: debt.height,
        generation: debt.generation,
        format: crate::runtime::draw::gva_resident_format(debt.format),
    }
}

#[cfg(feature = "backend-vulkan")]
fn release_gva(debt: GvaWritebackDebt) {
    crate::backend::vulkan::engine::note_resident_content_copied_out(&gva_identity(debt));
}

#[cfg(feature = "backend-vulkan")]
pub(crate) fn pay_key<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    key: WritebackKey,
) -> bool {
    match key {
        WritebackKey::Mapping(mapping_id) => {
            if let Some(debt) = state.pending_writebacks.take(mapping_id) {
                pay(state, host, mapping_id, debt, "wbdebt_paid_evicted");
            }
            true
        }
    }
}

/// Pay `mapping_id`'s owed frame and then wait for every guest-page write this
/// device has submitted — the whole obligation of a host-side reader or writer
/// of one named mapping's bytes, in one call.
///
/// The two halves are one obligation and are spelled as one function so a new
/// site cannot discharge half of it. A site that settles without paying reads
/// the frame *before* the one the guest's own driver last asked for; a site that
/// pays without settling reads the frame it just submitted and has not waited
/// for. Both are stale pixels and neither shows up as a refusal.
///
/// Free when nothing is owed and nothing is outstanding, which is the answer on
/// nearly every call: one emptiness check and one relaxed atomic load.
pub fn settle_for_mapping<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    site: crate::runtime::render_writeback::SettleSite,
) {
    pay_for_mapping(state, host, mapping_id);
    crate::runtime::render_writeback::settle_guest_writes(site);
}

/// [`settle_for_mapping`] for a caller that is **about to land the owed frame
/// itself**, over the same window, while preserving ranges the payment would
/// overwrite.
///
/// There is exactly one such caller and the distinction is not a nicety. A debt
/// is armed at a Store and names the resident that Store produced;
/// [`pay_for_mapping`] discharges it by writing that resident over the **whole**
/// window with no exclusions. `merge_guest_writes_into_pages` exists to put the
/// same resident into every page the guest did *not* write and keep the pages it
/// did — so paying first destroys exactly the bytes the merge was called to
/// preserve, one statement before the merge writes everything else back around
/// them. The guest's repaint is then gone and `t11sample_resident_merged`
/// reports success.
///
/// So the debt is **dropped**, not paid: what the caller is about to write is the
/// same surface's newer content at the same geometry — `write_bgra8_inner`
/// refuses on `GeometryMoved` if it is not — so the owed frame is superseded
/// rather than lost. The settle half still runs, because writes this device has
/// already submitted into these pages must land before the caller reads or
/// writes them, and that is true whoever is writing.
///
/// Counted, because a zero here says the two never co-occur on a workload and a
/// non-zero says how much guest painting the old order was throwing away.
pub fn supersede_for_mapping(
    state: &mut DeviceState,
    mapping_id: u32,
    site: crate::runtime::render_writeback::SettleSite,
) {
    if state.pending_writebacks.take(mapping_id).is_some() {
        crate::runtime::drain::note_store_route("wbdebt_superseded_by_skipping_write");
    }
    crate::runtime::render_writeback::settle_guest_writes(site);
}

/// [`settle_for_mapping`] for a caller that cannot name the mapping it is about
/// to touch, so it owes every debt.
pub fn settle_unnamed<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    site: crate::runtime::render_writeback::SettleSite,
) {
    pay_all(state, host);
    crate::runtime::render_writeback::settle_guest_writes(site);
}

/// Submit exactly the resources named by an asynchronous synchronize command.
///
/// The object list is the scope of the API operation; an unrelated host-valid
/// texture remains resident-authoritative. Completion belongs to the FIFO: the
/// transfers recorded here precede that packet's queue point, and its pending
/// stamp publishes only after that point completes. Waiting here would turn the
/// asynchronous command into a device-wide drain and then make the stamp wait a
/// second time for work already known complete.
pub fn submit_for_resources<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    object_ids: &[u32],
) {
    for &object_id in object_ids {
        pay_for_texture(state, host, task_id, object_id);
    }
}

/// [`settle_for_mapping`] over
/// [`crate::runtime::render_writeback::settle_guest_writes_unless_disjoint`].
///
/// The disjointness test narrows only the *wait*, never the payment: an owed
/// frame has not been submitted, so there is nothing outstanding for the test to
/// find disjoint from and a debt left unpaid here would be read straight past.
///
/// The page set is walked here rather than taken as a closure, and both of those
/// are deliberate. Walked *here* because the payment needs `state` mutably and
/// the disjointness test needs it shared, so a caller-supplied closure cannot
/// hold `state` across both. `DeviceState::mapping_reach_pages` because that is
/// the same function the writeback builds its own destination from, so the two
/// ends of the comparison are one rule rather than two spellings of it. It stays
/// lazy: `settle_guest_writes_unless_disjoint` runs the closure only when
/// something is outstanding.
pub fn settle_for_mapping_unless_disjoint<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    site: crate::runtime::render_writeback::SettleSite,
) {
    pay_for_mapping(state, host, mapping_id);
    let s = &*state;
    crate::runtime::render_writeback::settle_guest_writes_unless_disjoint(site, || {
        s.mapping_reach_pages(mapping_id)
    });
}

/// Count a reader that reaches guest bytes while a frame is owed and cannot pay
/// it, because it holds `DeviceState` immutably.
///
/// There is one — `draw::read_buffer_bytes_resolved`, the CPU
/// read of a *buffer's* guest bytes. A buffer and a type-11 render surface are
/// separate guest allocations, so this fires only where the two alias, and
/// aliasing across id namespaces is real rather than theoretical (see
/// [`crate::runtime::host_writes`]). The gap is therefore counted rather than
/// argued away: a boot reading `wbdebt_unpaid_buffer_read` above zero is a boot
/// where a buffer read *may* have seen a superseded surface frame, and that is
/// the signal to thread `&mut DeviceState` down to it.
///
/// A driven macos-13 sustained-animation boot puts `settle_buffer_guest_read` at
/// zero, which is the same call site counted for the waits it took.
pub fn note_unpaid_buffer_read(state: &DeviceState) {
    if !state.pending_writebacks.is_empty() {
        crate::runtime::drain::note_store_route("wbdebt_unpaid_buffer_read");
    }
}

/// Run the Store the debt stands for, now.
///
/// Everything the copy needs is resolved here and not at the arm — the identity
/// from the mapping's *current* generation, the page walk inside
/// `store_render_frame`. Two answers other than writing, and both release the
/// resident's `gpu_only_content` where they can, because that flag is what keeps
/// the reclaim off an image holding pixels nothing else has:
///
/// * **The guest superseded the frame.** `clear_host_valid` after the arm means
///   the guest wrote these pages itself, and landing an older frame on top of
///   its work is the write-ordering hazard `render_writeback`'s doc names
///   fourth. [`crate::runtime::resource_validity::licence_of`] is the existing
///   happens-before and it is read rather than re-derived.
/// * **The mapping's generation moved.** The guest remapped the surface, so the
///   identity this debt was armed under names a resident that is now an orphan,
///   and the pages it would be written into belong to something else. There is
///   no way to name that orphan from here — the current generation resolves to a
///   different identity — so its `gpu_only_content` outlives it and one image
///   leaks per occurrence. `wbdebt_generation_moved` is how a boot says how many;
///   a reading above single digits is the signal to carry the arm's whole
///   identity rather than the four integers that re-derive it.
#[cfg(feature = "backend-vulkan")]
fn pay<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    debt: WritebackDebt,
    route: &'static str,
) {
    let Some(entry) = state.mappings.get(&mapping_id) else {
        crate::runtime::drain::note_store_route("wbdebt_generation_moved");
        return;
    };
    let (map_generation, validity) = (entry.map_generation, entry.validity);
    if map_generation != debt.map_generation {
        crate::runtime::drain::note_store_route("wbdebt_generation_moved");
        return;
    }
    let identity = crate::runtime::present_identity::surface_identity(
        state,
        mapping_id,
        debt.width,
        debt.height,
    );
    if crate::runtime::resource_validity::licence_of(validity)
        == crate::runtime::resource_validity::WritebackLicence::Superseded
    {
        crate::runtime::drain::note_store_route("wbdebt_abandoned_guest_wrote");
        crate::backend::vulkan::engine::note_resident_content_copied_out(&identity);
        return;
    }
    crate::runtime::drain::note_store_route(route);
    if !crate::runtime::render_writeback::store_render_frame(
        state,
        host,
        mapping_id,
        &identity,
        debt.width,
        debt.height,
    ) {
        // `store_render_frame` reports its own loss on the failure channel; this
        // names the rail that owed it, because a debt paid late and refused is a
        // different investigation from a Store refused where it was issued.
        crate::observe::fail(format!(
            "wbdebt_pay_lost mapping={mapping_id} {}x{} reason=store_refused",
            debt.width, debt.height
        ));
        crate::backend::vulkan::engine::note_resident_content_copied_out(&identity);
    }
    crate::runtime::mapper::stamp_guest_write_gen(state, host, mapping_id);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GvaPaySite {
    Named,
    All,
}

#[cfg(feature = "backend-vulkan")]
impl GvaPaySite {
    fn route(self) -> &'static str {
        match self {
            Self::Named => "gvadebt_paid_named",
            Self::All => "gvadebt_paid_all",
        }
    }
}

/// Materialize one host-authoritative GVA resource into its retained transfer
/// backing. After explicit discard, synchronize lazily recreates that backing;
/// ordinary virtual-memory unmap does not participate in resource lifetime.
#[cfg(feature = "backend-vulkan")]
fn pay_gva<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    key: GvaResourceKey,
    debt: GvaWritebackDebt,
    site: GvaPaySite,
) -> bool {
    let identity = gva_identity(debt);
    let now = state.buffer_write_gen.stamp(key.task_id, key.texture_ref);
    if !now.quiet_since(debt.guest_write) {
        crate::runtime::drain::note_store_route("gvadebt_abandoned_guest_wrote");
        release_gva(debt);
        return true;
    }
    let Some(span) = u64::from(debt.row_stride).checked_mul(u64::from(debt.height)) else {
        crate::observe::fail(format!(
            "gvadebt_pay_lost task={} texture={} reason=span_overflow",
            key.task_id, key.texture_ref
        ));
        release_gva(debt);
        return true;
    };
    let generation = gva_resource_generation(state, host, key, debt.gva, span);
    let Some((backing_generation, backing_gva, backing_span, ordered)) =
        state.pending_writebacks.gva_resource_backing(key)
    else {
        state.pending_writebacks.restore_gva(key, debt);
        crate::runtime::drain::note_store_route(match site {
            GvaPaySite::Named => "gvadebt_named_unmapped",
            GvaPaySite::All => "gvadebt_all_unmapped",
        });
        if site == GvaPaySite::Named {
            crate::observe::fail(format!(
                "gvadebt_pay_blocked task={} texture={} reason=span_unresolved",
                key.task_id, key.texture_ref
            ));
        }
        return false;
    };
    if generation == 0
        || backing_generation != debt.generation
        || backing_gva != debt.gva
        || backing_span != span
    {
        crate::runtime::drain::note_store_route("gvadebt_generation_moved");
        release_gva(debt);
        return true;
    }
    let pages = crate::runtime::draw::StoreTargetPages::from_ordered(&ordered, span);
    let request = crate::runtime::draw::ColorRtRequest {
        texture_ref: key.texture_ref,
        target_gva: debt.gva,
        row_stride: debt.row_stride,
        width: debt.width,
        height: debt.height,
        format: debt.format,
        store_action: crate::contract::pass_action::MTL_STORE_ACTION_STORE,
        ..Default::default()
    };
    crate::runtime::drain::note_store_route(site.route());
    if let Err(reason) = crate::runtime::render_writeback::store_gva_frame(
        state,
        host,
        key.task_id,
        &identity,
        &request,
        key.texture_ref,
        Some(&pages),
    ) {
        crate::observe::fail(format!(
            "gvadebt_pay_lost task={} texture={} reason={reason}",
            key.task_id, key.texture_ref
        ));
        release_gva(debt);
    }
    true
}

/// [`pay`] on an arm with no Vulkan engine to owe a frame to.
///
/// Unreachable rather than merely unused: the only arm site is the type-11
/// surface Store in `draw::vulkan`, so the ledger is empty on this arm and both
/// callers return at their emptiness check before reaching here. It exists so
/// the reader-side helpers can be one set of functions on both arms instead of
/// two spellings the settle sites would have to choose between.
#[cfg(not(feature = "backend-vulkan"))]
fn pay<M: HostMemory + HostOps>(
    _state: &mut DeviceState,
    _host: &mut M,
    _mapping_id: u32,
    _debt: WritebackDebt,
    _route: &'static str,
) {
}

#[cfg(not(feature = "backend-vulkan"))]
fn pay_gva<M: HostMemory + HostOps>(
    _state: &mut DeviceState,
    _host: &mut M,
    _key: GvaResourceKey,
    _debt: GvaWritebackDebt,
    _site: GvaPaySite,
) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The coalescing the rail exists for, at the container: a second arm into
    /// one mapping replaces the first rather than queueing beside it, so N
    /// Stores between two reads cost one copy and not N.
    #[test]
    fn a_second_arm_into_one_mapping_replaces_the_first() {
        let mut pending = PendingWritebacks::default();
        assert_eq!(pending.arm(7, 1920, 1080, 3), None);
        assert_eq!(pending.arm(7, 1920, 1080, 3), None);
        assert_eq!(pending.len(), 1, "one mapping owes one frame");
        let debt = pending.take(7).expect("mapping 7 owes a frame");
        assert_eq!(debt.seq, 1, "the later Store is the one owed");
        assert!(pending.is_empty());
    }

    /// Geometry travels with the debt, because the payment writes at the
    /// geometry the Store was taken at and the mapping may have been re-declared
    /// since.
    #[test]
    fn a_debt_carries_the_geometry_its_store_was_taken_at() {
        let mut pending = PendingWritebacks::default();
        assert_eq!(pending.arm(4, 800, 600, 11), None);
        let debt = pending.get(4).expect("mapping 4 owes a frame");
        assert_eq!(
            (debt.width, debt.height, debt.map_generation),
            (800, 600, 11)
        );
    }

    /// The bound is the container's, and it hands the caller the mapping that
    /// has to be paid rather than dropping a frame to stay under it.
    #[test]
    fn arming_past_the_bound_evicts_the_oldest_and_says_so() {
        let mut pending = PendingWritebacks::default();
        for id in 0..MAX_DEBTS as u32 {
            assert_eq!(pending.arm(id, 64, 64, 1), None, "under the bound");
        }
        assert_eq!(pending.len(), MAX_DEBTS);
        let evicted = pending.arm(MAX_DEBTS as u32, 64, 64, 1);
        assert_eq!(
            evicted,
            Some(WritebackKey::Mapping(0)),
            "the oldest arm is the one handed back"
        );
        assert_eq!(
            pending.len(),
            MAX_DEBTS + 1,
            "the named debt is still owed until the caller pays it"
        );
        assert!(
            pending.take(0).is_some(),
            "and paying it is what brings the ledger back under the bound"
        );
        assert_eq!(pending.len(), MAX_DEBTS);
    }

    /// Re-arming a mapping already at the head of a full ledger must not evict:
    /// it is a replacement and the entry count does not grow.
    #[test]
    fn re_arming_a_held_mapping_never_evicts() {
        let mut pending = PendingWritebacks::default();
        for id in 0..MAX_DEBTS as u32 {
            assert_eq!(pending.arm(id, 64, 64, 1), None);
        }
        assert_eq!(
            pending.arm(0, 64, 64, 1),
            None,
            "a replacement makes no room"
        );
        assert_eq!(pending.len(), MAX_DEBTS);
    }

    /// Age order is arm order and not mapping id, because `pay_all` walks it and
    /// the oldest owed frame is the one most likely to be read.
    #[test]
    fn mappings_come_back_in_arm_order() {
        let mut pending = PendingWritebacks::default();
        assert_eq!(pending.arm(9, 1, 1, 1), None);
        assert_eq!(pending.arm(2, 1, 1, 1), None);
        assert_eq!(pending.arm(5, 1, 1, 1), None);
        assert_eq!(pending.mappings_by_age(), vec![9, 2, 5]);
    }

    fn gva_debt(generation: u64) -> GvaWritebackDebt {
        GvaWritebackDebt {
            gva: 0x4000,
            row_stride: 256,
            width: 64,
            height: 64,
            format: crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            generation,
            guest_write: Default::default(),
            seq: 0,
        }
    }

    /// The resource reference, not the GVA, owns coherence. Reusing the same
    /// resource for another Store replaces its debt exactly as repeated Stores
    /// into one IOSurface do.
    #[test]
    fn a_second_gva_store_on_one_resource_replaces_the_first() {
        let mut pending = PendingWritebacks::default();
        let key = GvaResourceKey {
            task_id: 3,
            texture_ref: 19,
        };
        assert_eq!(pending.arm_gva(key, gva_debt(7)), None);
        let previous = pending.arm_gva(key, gva_debt(8));
        assert_eq!(previous.map(|debt| debt.generation), Some(7));
        assert_eq!(pending.len(), 1);
        assert_eq!(pending.get_gva(key).map(|debt| debt.generation), Some(8));
    }

    /// GVA resources have protocol lifetime, not an arbitrary ledger capacity.
    /// Holding more than the anonymous-surface coalescing bound must not invent
    /// a transfer or drop an older resource's host-authoritative frame.
    #[test]
    fn gva_resources_are_not_evicted_by_the_surface_debt_bound() {
        let mut pending = PendingWritebacks::default();
        for texture_ref in 1..=(MAX_DEBTS as u32 + 8) {
            let key = GvaResourceKey {
                task_id: 2,
                texture_ref,
            };
            pending.ensure_gva_resource(
                key,
                u64::from(texture_ref) << 16,
                4096,
                Some(vec![u64::from(texture_ref) << 12]),
            );
            assert_eq!(pending.arm_gva(key, gva_debt(texture_ref.into())), None);
        }
        assert_eq!(pending.len(), MAX_DEBTS + 8);
        assert_eq!(pending.gvas_by_age().len(), MAX_DEBTS + 8);
    }

    /// Ordinary virtual-memory bookkeeping does not retarget a live resource.
    /// A repeated prepare with a different walk keeps the original transfer
    /// backing until the protocol explicitly discards it.
    #[test]
    fn a_live_resource_retains_its_backing_until_discard() {
        let mut pending = PendingWritebacks::default();
        let key = GvaResourceKey {
            task_id: 3,
            texture_ref: 19,
        };
        let generation = pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0x9000]));
        assert_eq!(
            pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0xa000])),
            generation
        );
        assert_eq!(&*pending.gva_resource_backing(key).unwrap().3, &[0x9000]);

        assert_eq!(pending.discard_gva_resources(3, &[19]), 1);
        assert!(pending.gva_resource_backing(key).is_none());
        assert_eq!(
            pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0xa000])),
            generation,
            "discard replaces the transfer backing, not the host texture"
        );
        assert_eq!(&*pending.gva_resource_backing(key).unwrap().3, &[0xa000]);
    }

    /// Delete is the resource lifetime boundary. Reusing the same task-local
    /// reference after delete receives a new host-texture identity.
    #[test]
    fn deleting_and_recreating_a_resource_changes_its_generation() {
        let mut pending = PendingWritebacks::default();
        let key = GvaResourceKey {
            task_id: 3,
            texture_ref: 19,
        };
        let first = pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0x9000]));
        assert!(pending.retire_gva_resource(key).0);
        let second = pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0xa000]));
        assert_ne!(first, second);
    }

    /// A guest validity transition after the Store makes guest memory newer
    /// than the held resident. The debt remains available for an orderly
    /// abandon, but it must immediately stop licensing host-resident reads.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn a_guest_write_revokes_gva_resident_authority() {
        let mut state = DeviceState::new(crate::model::DeviceId::default(), 12);
        let key = GvaResourceKey {
            task_id: 4,
            texture_ref: 12,
        };
        let debt = gva_debt(99);
        let _ = state.pending_writebacks.arm_gva(key, debt);
        let identity = gva_identity(debt);
        assert!(gva_resident_authoritative(&state, &identity));
        state
            .buffer_write_gen
            .note_write(key.task_id, key.texture_ref);
        assert!(!gva_resident_authoritative(&state, &identity));
        assert!(state.pending_writebacks.get_gva(key).is_some());
    }

    /// A synchronize list is a scope, not merely a trigger. Publishing one
    /// object must leave an unrelated resource host-authoritative.
    #[test]
    fn asynchronous_resource_synchronization_submits_only_named_objects() {
        let mut state = DeviceState::new(crate::model::DeviceId::default(), 12);
        let mut host = crate::runtime::FakeHost::new();
        assert_eq!(state.pending_writebacks.arm(7, 64, 64, 1), None);
        assert_eq!(state.pending_writebacks.arm(8, 64, 64, 1), None);
        submit_for_resources(&mut state, &mut host, 1, &[7]);
        assert!(state.pending_writebacks.get(7).is_none());
        assert!(state.pending_writebacks.get(8).is_some());
    }
}
