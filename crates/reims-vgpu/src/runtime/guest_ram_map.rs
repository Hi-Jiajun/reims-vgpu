//! The process's imports of guest RAM: built once from the shim's spans, and
//! the only place a guest physical address becomes a bindable reference.
//!
//! # What this replaces
//!
//! The dma-buf rail had a cache here, and it had to: `UDMABUF_CREATE_LIST`
//! walked every page, took a kernel reference on each, and cost enough that a
//! digest-bucketed LRU bounded by pinned bytes was worth its own module.
//!
//! Under the host-pointer model there is nothing left to cache.
//! [`crate::runtime::guest_ram::GuestRamImport::slice`] is a range check.
//! What *is* worth holding is the small thing the cache was built around: the
//! imports themselves, made once at first use and held for the VM's lifetime.
//! This module is that, and it is a sorted `Vec` of a dozen or so entries rather
//! than a cache with an eviction policy.
//!
//! A RAMBlock is imported in **chunks** rather than whole, because a driver has
//! been measured truncating a multi-gigabyte `allocationSize` to 32 bits on this
//! path and reading back garbage past the truncation — see
//! [`crate::backend::vulkan::caps::host_pointer::IMPORT_SPAN_CEILING`]. Chunking
//! needed nothing else to change: a window has always resolved against whichever
//! import backs its GPA, and one straddling two of them has always grouped into
//! two `VkBuffer` sources, because a RAMBlock boundary could already split one.
//!
//! # Why the imports are built here and not at device create
//!
//! The backend measures the granularity; the runtime holds the
//! [`HostOps`](crate::runtime::HostOps) that can say where guest RAM lives.
//! Neither side has both, and the device context deliberately does not take a
//! host — see the module doc on [`crate::qemu::host_ops`] for why the runtime
//! keeps it. So the granularity is published by the backend through
//! [`crate::runtime::guest_ram::latch_import_limits`] — together with the
//! largest import that backend's heaps could hold — and the spans are fetched
//! here, on the first guest-memory reference of a boot.
//!
//! Building lazily rather than eagerly also gets the ordering right for free:
//! the device exists before any guest command is decoded, so the granularity is
//! always published by the time the first reference is asked for.
//!
//! # What a refusal means
//!
//! Every refusal here puts the whole boot on the copying rails for the
//! addresses it covers, so none of them is a slow path and none may be silent.
//! The one *expected* refusal is [`MapRefusal::NoBackendImport`](crate::runtime::guest_ram_map::MapRefusal::NoBackendImport): a host without
//! the extension, or an operator who set
//! [`crate::config::GUEST_IMPORT`](crate::config::GUEST_IMPORT) off. That one is a
//! statement about the host rather than a loss, so it is reported once on the
//! off channel rather than as a failure per reference.

use crate::backend::Backend as _;
use crate::runtime::guest_ram::{
    granularity, import_budget, import_span_max, GuestRamError, GuestRamImport, GuestRamRegion,
    GuestRef, ImportId,
};
use crate::runtime::host::{GuestRamRegionsError, HostOps};
use std::sync::Arc;

/// Why a guest physical address did not become a bindable reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapRefusal {
    /// No backend published an import granularity: this host cannot import
    /// guest RAM, or an operator asked it not to. Expected, and the state every
    /// copying rail exists for.
    NoBackendImport,
    /// The shim could not say where guest RAM lives. Carries the check that
    /// refused.
    HostRefused(GuestRamRegionsError),
    /// The shim answered, but no span survived being bounded to the granularity
    /// — every region was empty, unmapped, malformed, or shorter than one
    /// granule. Distinct from [`Self::HostRefused`] because the host answered
    /// fine and it is our own bound that rejected every span.
    NoUsableRegion { spans: usize },
    /// The spans are importable and this guest is larger than the roomiest heap
    /// on the host's GPU, so nothing may import any of them.
    ///
    /// # Why the whole map refuses rather than the block that does not fit
    ///
    /// An import is a `VkDeviceMemory` charged to one heap, and a submission
    /// that names it makes the driver keep all of it resident. On a part whose
    /// heaps are a fraction of the guest — an APU with a few gigabytes of
    /// carve-out against a `-m 16G` guest — the kernel refuses to validate the
    /// allocation and the submission fails, which arrives as a **lost device**
    /// and a dead guest rather than as a slow rail. That has been reported from
    /// the field on `radv`/`amdgpu` (`Not enough memory for command submission`,
    /// then a lost context), and the reporter's own fix was to set
    /// [`crate::config::GUEST_IMPORT`] off by hand. This makes the device reach the
    /// same state without being told to.
    ///
    /// Refusing the whole map rather than the oversized block is what makes it
    /// safe: the copying rails are selected by a page having no [`GuestRef`], and
    /// a *partial* import would leave the writeback paths holding references
    /// into one RAMBlock and none into another, which is a hard error at those
    /// sites and not a fallback. All or nothing keeps the boot on the one arm
    /// that is tested end to end.
    ///
    /// The comparison is against the sum, because every import is live at once
    /// for the VM's lifetime and a submission may name any of them.
    ///
    /// # Its relationship to the per-import check, which is the exact one
    ///
    /// The backend publishes this budget as the roomiest heap an import can be
    /// *charged to* — the same population of memory types
    /// `caps::memory_topology::select_memory_type` will choose from, since every
    /// import goes through it carrying one class's required flags. So a sum that
    /// passes here has a heap that each individual chunk fits, and the exact
    /// per-allocation check at the pick — which refuses rather than making a
    /// call Vulkan declares invalid — agrees with this one by construction
    /// rather than by coincidence. Publishing the maximum over *every* heap
    /// instead, which this once did, breaks that: a part whose device-local heap
    /// is twice its host-visible heap passes here with room to spare and then
    /// refuses at every pick, which is the partial import this refusal exists to
    /// prevent.
    ///
    /// This is a heap-*capacity* test and not a residency one, so it is a lower
    /// bound: a host that passes it can still be too full to import. It catches
    /// the direction that has been seen to kill a guest.
    ///
    /// # What would give such a host the fast rail back
    ///
    /// Not this refusal, which only makes the host work. A RAMBlock is now
    /// imported in chunks — for an unrelated reason, a driver truncating
    /// `allocationSize` — and two of the three things that stood in the way of
    /// using those chunks for *residency* have gone with it: the split exists,
    /// and the lookup is a binary search rather than a linear `find`, so a
    /// hundred imports cost what two did.
    ///
    /// What is left is the part that was always the hard one. Chunks are still
    /// imported eagerly and all at once, and the sum is still what this compares,
    /// so the refusal fires exactly where it did. Making a small-heap host work
    /// means importing a chunk on first reference into it and *releasing* chunks
    /// again — against this module's standing advice, which is sound only while
    /// every import fits. That cannot be measured on a host whose roomiest heap
    /// is four times its guest.
    ImportExceedsHeap { needed: u64, budget: u64 },
    /// The address is not inside any imported span. Guest RAM the GPU can reach
    /// exists, and this address is not in it — a device MMIO address, a hole,
    /// or a page the guest named that this machine does not back.
    GpaNotInAnyImport { gpa: u64 },
    /// The address is in a span, and the length asked for leaves it. Carries the
    /// bound's own reason so the check that refused keeps its name.
    OutsideImport(GuestRamError),
    /// A page list that is not one GPA-contiguous stretch.
    ///
    /// Not a statement that the pages are un-importable — they are all inside
    /// one RAMBlock and each is nameable. It is a statement about the *bind*: a
    /// `VkBuffer` range and a Metal buffer offset are each one offset and one
    /// length, so a surface assembled from four stretches is four of them, and
    /// no consumer takes several yet. Named and counted because how often it
    /// fires is what says whether widening them is worth doing.
    ///
    /// `runs` is what says *how much* widening would cost, and it is the number
    /// to read before building it. "Scattered" is one word for both a window in
    /// two stretches — where a second bind is obviously worth it — and a window
    /// in five hundred, where each run is a couple of pages and the region list
    /// starts to rival the copy it replaces. `pages` alone cannot tell those
    /// apart, and a count of *refusals* tells them apart even less: both read as
    /// one line here. Sampled at the point of refusal, so it bands the reach
    /// actually requested rather than the reach some other rail asked for.
    ///
    /// # What it measured, on an x86 guest
    ///
    /// A driven boot — Safari window drag, 25 s, PCI attach — put **every**
    /// window at almost exactly four pages per run, across three orders of
    /// magnitude of size: 2025 pages in 507 runs for each 1920x1080 writeback,
    /// 813/204, 630/158, 588/147, 256/65, 128/32, 45/12. The ratio holds
    /// because it is not a ratio: the runs are 16 KiB each, and the guest backs
    /// a surface in 16 KiB physically-contiguous granules that are unrelated to
    /// each other. Four 4 KiB x86 pages is what one of those granules looks
    /// like from this side.
    ///
    /// Two consequences worth carrying, because both contradict the obvious
    /// guess. Scattering is **not** a fragmentation artifact that a longer
    /// uptime or a quieter guest would improve — it is the allocator's
    /// granularity, so it is the steady state. And the run count scales with
    /// the surface, so the widening this field exists to price is ~500 ranges
    /// for a full-screen flush and not the handful the word "scattered"
    /// suggests.
    Scattered {
        pages: usize,
        runs: usize,
        first: u64,
    },
}

impl crate::observe::Decline for MapRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::NoBackendImport => "guest_ram_map_no_backend_import",
            Self::HostRefused(_) => "guest_ram_map_host_refused",
            Self::NoUsableRegion { .. } => "guest_ram_map_no_usable_region",
            Self::ImportExceedsHeap { .. } => "guest_ram_map_import_exceeds_heap",
            Self::GpaNotInAnyImport { .. } => "guest_ram_map_gpa_not_in_any_import",
            Self::Scattered { .. } => "guest_ram_map_scattered",
            // The inner reason is the diagnosis; this wrapper only says where
            // it happened, so it forwards rather than adding a slug of its own.
            Self::OutsideImport(inner) => inner.slug(),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::NoBackendImport => Vec::new(),
            Self::HostRefused(inner) => {
                let mut f = vec![("host_reason", inner.slug().to_string())];
                f.extend(crate::observe::Decline::fields(inner));
                f
            }
            Self::NoUsableRegion { spans } => vec![("spans", spans.to_string())],
            Self::ImportExceedsHeap { needed, budget } => vec![
                ("needed_mb", (needed >> 20).to_string()),
                ("budget_mb", (budget >> 20).to_string()),
            ],
            Self::GpaNotInAnyImport { gpa } => vec![("gpa", format!("{gpa:#x}"))],
            Self::Scattered { pages, runs, first } => vec![
                ("pages", pages.to_string()),
                ("runs", runs.to_string()),
                ("first", format!("{first:#x}")),
            ],
            Self::OutsideImport(inner) => inner.fields(),
        }
    }
}

crate::observe::decline_display!(MapRefusal);

/// The greppable event class for this module's refusals.
const EVENT: &str = "guest_ram_map";

/// The imports this process holds, or the refusal that stopped it building any.
///
/// Resolved once and then read. A `Mutex` rather than a `OnceLock` because a
/// device recreate must be able to drop the imports: the backend's handles die
/// with the device, and an import whose identity outlived them would let a
/// stale [`crate::runtime::guest_ram::GuestSlice`] resolve against a
/// `VkDeviceMemory` that no longer exists.
static MAP: std::sync::Mutex<Option<Resolved>> = std::sync::Mutex::new(None);

#[derive(Debug)]
struct Resolved {
    /// One per usable RAMBlock span, in the order the shim reported them.
    /// Ordinary machines have one or two.
    imports: Vec<Arc<GuestRamImport>>,
    /// Set when the resolution refused, so the next reference does not re-ask
    /// the shim for an answer that will not change. A refusal here is about the
    /// host and the granularity, both of which are fixed for the device's life.
    refusal: Option<MapRefusal>,
}

impl Resolved {
    /// Turn one guest physical address and length into a bindable reference.
    ///
    /// The single implementation, so the one-span and the whole-window entry
    /// points cannot disagree about which import owns a GPA or which refusal a
    /// miss earns. Takes no lock of its own — the caller is already inside
    /// [`with_map`], which is what lets a scattered window resolve every run
    /// under one acquisition.
    ///
    /// [`Self::refusal`] is **not** re-checked here: an entry point asks it once
    /// before walking, and asking again per run would emit the same standing
    /// refusal N times for one window.
    fn reference(&self, gpa: u64, len: u64) -> Result<GuestRef, MapRefusal> {
        // Binary search, not a linear scan. The imports are sorted by
        // `gpa_base`, so the last one whose base is at or below `gpa` is the
        // only one that can contain it — `partition_point` names that index and
        // `contains_gpa` still decides, so an address in a hole between two
        // imports is refused exactly as before.
        //
        // A scan was right while a machine had one or two imports. Chunking a
        // RAMBlock at the span ceiling makes it eight to a dozen on an ordinary
        // guest, and this runs once per run of a scattered window — 9 to 32 runs
        // per bind, thousands of binds a second — so the growth would land in
        // the hot path. This makes the count stop mattering instead of trading
        // one host's correctness for another's throughput.
        let import = self
            .imports
            .partition_point(|i| i.gpa_base().is_some_and(|base| base <= gpa))
            .checked_sub(1)
            .map(|last| &self.imports[last])
            .filter(|i| i.contains_gpa(gpa))
            .ok_or(MapRefusal::GpaNotInAnyImport { gpa })
            .map_err(report_once)?;
        // `slice_for_gpa` emits its own named refusal on the fail channel, so
        // the wrapper forwards the reason rather than adding a second line.
        let slice = import
            .slice_for_gpa(gpa, len)
            .map_err(MapRefusal::OutsideImport)?;
        GuestRef::new(Arc::clone(import), slice).map_err(MapRefusal::OutsideImport)
    }

    /// The exclusive end GPA of the import backing `gpa`, or `None` if nothing
    /// backs it.
    ///
    /// Exists so a caller can split a contiguous guest stretch at the seam
    /// between two imports instead of being refused at it — see
    /// [`references_for_runs`]. Deliberately not a public entry point: an import
    /// boundary is this module's own bookkeeping, and the only thing outside it
    /// may do with one is stop at it.
    fn import_end(&self, gpa: u64) -> Option<u64> {
        self.imports
            .partition_point(|i| i.gpa_base().is_some_and(|base| base <= gpa))
            .checked_sub(1)
            .map(|last| &self.imports[last])
            .filter(|i| i.contains_gpa(gpa))
            .and_then(|i| i.gpa_base().map(|base| base + i.len()))
    }
}

/// Forget every import.
///
/// Called when the backend tears its device down. The next reference rebuilds,
/// against fresh identities, so nothing made before the teardown resolves after
/// it — see [`crate::runtime::guest_ram::ImportId`] for why that matters.
pub fn reset() {
    *MAP.lock().unwrap_or_else(|p| p.into_inner()) = None;
}

/// How many RAMBlock spans the shim reported, and how many bytes they cover.
///
/// The denominator for the backend's *imported* count. A backend imports a span
/// at its first reference and not before, so "one imported" means one of these
/// has been touched — which on a two-span machine is a workload fact, not a
/// defect. Reporting the count alone cannot tell those apart, which is why the
/// census line carries both.
///
/// Counts rather than clones: this runs once a census window, and cloning an
/// `Arc` per span to take a length would be a refcount touch per span per
/// second for a number that does not change after the first reference.
///
/// `(0, 0)` before the first reference of a boot and on a host that cannot
/// import, which is the same reading the census suppresses.
pub fn span_census() -> (usize, u64) {
    MAP.lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
        .map(|r| (r.imports.len(), r.imports.iter().map(|i| i.len()).sum()))
        .unwrap_or((0, 0))
}

/// Every import this process holds, for a backend that needs to create or
/// release its device-side handles.
///
/// Empty before the first reference of a boot and on a host that cannot import.
pub fn imports() -> Vec<Arc<GuestRamImport>> {
    MAP.lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
        .map(|r| r.imports.clone())
        .unwrap_or_default()
}

/// One import this process holds, described the way a provider's `HostRegion`
/// registration needs it.
///
/// # What this is, and what it is not
///
/// A **read-only projection**, not a registration. Reading one hands nothing to
/// a backend, takes no lease, and changes no state: it is the same import list
/// [`imports`] returns, named field by field.
///
/// It exists because the two coordinates had no meeting point. A provider's
/// `HostRegion` wants `{host_pointer, length, page_size}` plus an identity it can
/// derive a lease from, and this module holds all four — while nothing in the
/// tree can currently say "these are the bytes this device imports". The
/// provider-side registration itself does not exist yet; this is the projection
/// it will be written against, and it is deliberately answerable on any host,
/// GPU or not.
///
/// # Why a type of its own rather than [`GuestRamRegion`]
///
/// [`GuestRamRegion`] is the *shim's* shape: what QEMU reported, before this
/// module bounded any of it. What an import *covers* is that answer after
/// chunking and trimming, and it carries two things the shim never said — the
/// import's identity ([`ImportId`], process-monotonic and never reused, so it is
/// the one thing a lease id can be derived from without inventing a second id
/// space) and the granularity the registration must declare as `page_size`.
/// Neither belongs in the ABI-mirrored `GuestRamRegion`, and dropping them would
/// leave the projection unable to answer the question it was added for. So the
/// shape lives here, beside the map it is a view of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostRegion {
    /// Identity of the import this projects — [`GuestRamImport::id`]. Distinct
    /// for every region, including two chunks of one RAMBlock, and never
    /// reused: a lease derived from it cannot resolve against the *replacement*
    /// import over the same RAMBlock after a device recreate.
    pub import: ImportId,
    /// First guest physical address this import covers, or `None` for an import
    /// with no linear GPA coordinate — the packed-alias shape
    /// ([`GuestRamImport::new_host_allocation`]). Every RAMBlock import has a
    /// base, and every import in this projection is one today; the `Option` is
    /// how the underlying type answers, carried rather than unwrapped so a
    /// future alias registered here stays describable instead of a panic.
    pub gpa_base: Option<u64>,
    /// Host virtual address covered, after any trim. This is the address the
    /// import call was given, and never a subrange of it.
    pub host_va: u64,
    /// Bytes covered, in both address spaces. Always a multiple of
    /// [`Self::page_size`].
    pub len: u64,
    /// The backend's import granularity this import was built against, which is
    /// the only legal `page_size` for a window over it — 0 and non-powers-of-two
    /// are refused at the registration, not clamped.
    pub page_size: u64,
}

/// Project every guest RAM import this process holds into [`HostRegion`] shape.
///
/// A projection and not a registration: nothing is handed to a backend, no
/// lease is taken, and no state is written. Callers that only want to *read* an
/// established map without taking the import should use [`imports`] instead.
///
/// # When it runs, and what that does to [`warm`]
///
/// Resolution here is the same one a reference takes and the same one [`warm`]
/// takes, so calling this first is `warm`'s first half: the import is
/// established, and the guest's first draw does not pay for it — including the
/// driver's page pinning, which is seconds on a large guest. Calling it again
/// changes nothing, and the shim is still asked exactly once per boot.
///
/// The guard is [`warm`]'s guard, for [`warm`]'s reason. Before a backend
/// publishes a granularity this returns an empty projection rather than
/// resolving: `resolve` answers `NoBackendImport` when there is no granularity,
/// that answer is latched in `MAP` for the rest of the boot, and a
/// projection taken one instant too early would therefore turn a host that can
/// import into one that refuses every window — reported, worse, as a host
/// lacking the extension.
///
/// An empty list has two readings, both correct and both boring: nothing has
/// been resolved yet, or the host cannot import. They are told apart by
/// [`standing_refusal`], which is the question that has an answer.
///
/// # Errors
///
/// The shim's own refusal, carried as itself. A host without the
/// `guest_ram_regions` callback — every pre-v17 shim and every fixture host —
/// answers [`GuestRamRegionsError::CallbackMissing`], and this hands that back
/// rather than folding it into an empty list. "This host cannot say where guest
/// RAM lives" and "this host has imported nothing yet" are different things to
/// go fix, and a projection whose whole job is to be compared against a
/// registration is the wrong place to lose the difference.
///
/// The other latched refusals ([`MapRefusal::NoUsableRegion`],
/// [`MapRefusal::ImportExceedsHeap`], [`MapRefusal::NoBackendImport`]) are not
/// host errors and project as the empty list they describe; a caller that needs
/// the reason asks [`standing_refusal`].
pub fn host_regions<H: HostOps + ?Sized>(
    host: &mut H,
) -> Result<Vec<HostRegion>, GuestRamRegionsError> {
    if granularity().is_none() {
        return Ok(Vec::new());
    }
    with_map(host, |resolved| {
        if let Some(MapRefusal::HostRefused(why)) = resolved.refusal {
            return Err(why);
        }
        Ok(resolved
            .imports
            .iter()
            .map(|import| HostRegion {
                import: import.id(),
                gpa_base: import.gpa_base(),
                host_va: import.host_base() as u64,
                len: import.len(),
                page_size: import.align(),
            })
            .collect())
    })
}

/// One import this process holds, bounded to the shape a registration may
/// name.
///
/// # Why a second type, when the projection is already registration-shaped
///
/// [`HostRegion`] is the projection: what this process holds, field by field.
/// This is that same range after the two bounds a registration imposes on
/// itself have been made *explicit* — the covered base is a whole number of
/// `page_size` granules from the region's start, and the covered length is a
/// whole number of granules.
///
/// The projection satisfies both today, but nothing in it says so: the fields
/// are plain `u64`s, and the one value a reader could compare them against, the
/// shim's `host_va` in [`GuestRamRegion`], is not rounded at all. A registration
/// that names an unaligned base or a partial granule is refused outright, so the
/// step between "these are the bytes we hold" and "this is the range that may be
/// registered" needs a name, one place for the arithmetic, and tests that run on
/// a host with no GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegistrationCandidate {
    /// This region's position in the projection it came from, so a refusal or a
    /// log line names the same element the reader can count to.
    ///
    /// The projection's own index rather than the candidate list's: a region
    /// that produced no candidate must not renumber the ones that did.
    pub region_index: usize,
    /// The import this candidate registers — [`HostRegion::import`], carried so
    /// a lease can be derived from an identity that is process-monotonic and
    /// never reused.
    pub import: ImportId,
    /// First guest physical address the *candidate* covers: [`HostRegion::gpa_base`]
    /// moved forward by however far the base had to be rounded, and `None` for
    /// an import with no linear GPA coordinate — the packed-alias shape.
    pub gpa_base: Option<u64>,
    /// The projected host base, before rounding.
    pub host_va: u64,
    /// The projected length, before rounding.
    pub len: u64,
    /// The granularity the import was built against, which is the only legal
    /// `page_size` for a window over it.
    pub page_size: u64,
    /// `host_va` rounded **up** to the next whole granule.
    ///
    /// Up rather than down because a registration covers bytes it names: a base
    /// rounded down would start before the region, which is the range the
    /// provider refuses rather than truncates. Never past `host_va + len` either
    /// — a base that lands on the region's own end leaves no candidate, below.
    pub aligned_base: u64,
    /// Whole granules from [`Self::aligned_base`] to the region's own end, i.e.
    /// that end rounded **down**. At least one granule, or the region produces
    /// no candidate at all.
    pub aligned_len: u64,
}

/// Why a projected region could not become a registration candidate.
///
/// One variant per distinct check, so a line in the fail log says which bound
/// refused rather than that something about a registration did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegistrationRefusal {
    /// The region's `page_size` is zero or not a power of two. Every rounding
    /// below is a mask, and the provider refuses the same input with its own
    /// `InvalidHostRegionPageSize`; refused rather than clamped, because a
    /// registration at an invented granularity is a window the provider and
    /// this side would disagree about.
    PageSizeNotPowerOfTwo { region_index: usize, page_size: u64 },
    /// `host_va + len` leaves the address space, so the region has no end a
    /// registration could be bounded to.
    ///
    /// A projection cannot be in this shape — [`GuestRamImport::new`] does this
    /// check when it builds the import — so this fires on a hand-built value,
    /// which is exactly what makes it worth naming.
    RegionLeavesAddressSpace {
        region_index: usize,
        host_va: u64,
        len: u64,
    },
}

impl crate::observe::Decline for RegistrationRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::PageSizeNotPowerOfTwo { .. } => {
                "guest_ram_registration_page_size_not_power_of_two"
            }
            Self::RegionLeavesAddressSpace { .. } => {
                "guest_ram_registration_region_leaves_address_space"
            }
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match *self {
            Self::PageSizeNotPowerOfTwo {
                region_index,
                page_size,
            } => vec![
                ("region_index", region_index.to_string()),
                ("page_size", page_size.to_string()),
            ],
            Self::RegionLeavesAddressSpace {
                region_index,
                host_va,
                len,
            } => vec![
                ("region_index", region_index.to_string()),
                ("host_va", format!("{host_va:#x}")),
                ("len", len.to_string()),
            ],
        }
    }
}

crate::observe::decline_display!(RegistrationRefusal);

/// Bound a projection to what a registration may name, or say why it cannot.
///
/// Pure: no host, no lock, no state, nothing latched. [`host_regions`] answers
/// what this process holds; this answers which of those ranges is registerable,
/// and it answers with one descriptor per region rather than a bool because the
/// caller has to hand each one over.
///
/// # Both bounds move inward
///
/// The base rounds up and the end rounds down, so every candidate sits inside
/// the region it came from: `aligned_base >= host_va` and `aligned_base +
/// aligned_len <= host_va + len`. Rounding outward is the tempting mistake — it
/// keeps all the bytes the region covers — and it is the one that cannot be
/// registered, because a window past the end of the registration is refused
/// rather than truncated.
///
/// A region whose end lands in a partial granule, or which is shorter than one
/// granule to begin with, yields **no** candidate rather than an illegal one:
/// such a range cannot be expressed as a registration, and inventing the
/// difference here would put a window over bytes this import does not own. The
/// regions around it are unaffected, and their `region_index` values stay the
/// projection's, so a dropped region is visible from the list alone.
///
/// Over a projection from [`host_regions`] both roundings are no-ops — every
/// import was trimmed to its granularity when it was built — which is what makes
/// this testable against the real projection and not only against hand-written
/// shapes. The guards are for the caller that is *not* handing over a projection,
/// and the one that matters at wiring time is exactly that: passing the shim's
/// raw `host_va` instead of the import's already-trimmed base.
///
/// # Errors
///
/// [`RegistrationRefusal`]. Reported by the caller as
/// `Emit::decline("guest_ram_registration", &refusal)` rather than emitted here:
/// this is a pure function, and a test of it should not have to read the fail
/// log to learn what it answered.
pub fn registration_candidates(
    regions: &[HostRegion],
) -> Result<Vec<RegistrationCandidate>, RegistrationRefusal> {
    let mut candidates = Vec::with_capacity(regions.len());
    for (region_index, region) in regions.iter().enumerate() {
        let page_size = region.page_size;
        if page_size == 0 || !page_size.is_power_of_two() {
            return Err(RegistrationRefusal::PageSizeNotPowerOfTwo {
                region_index,
                page_size,
            });
        }
        let end = region.host_va.checked_add(region.len).ok_or(
            RegistrationRefusal::RegionLeavesAddressSpace {
                region_index,
                host_va: region.host_va,
                len: region.len,
            },
        )?;
        // Checked, because a base in the last granule of the address space has
        // no representable aligned form at all. That reads the same way as a
        // region with no whole granule in it — no candidate — rather than as a
        // second refusal: the check below would skip it too, and a wrapped base
        // is the one answer that must not be produced.
        let aligned_base = match region.host_va.checked_add(page_size - 1) {
            Some(last) => last & !(page_size - 1),
            None => continue,
        };
        let aligned_len = (end & !(page_size - 1)).saturating_sub(aligned_base);
        if aligned_len < page_size {
            continue;
        }
        candidates.push(RegistrationCandidate {
            region_index,
            import: region.import,
            // Moved with the base: the candidate's first byte is the first
            // aligned one, and the GPA naming it is the import's own base plus
            // however far the base had to move. Both coordinates came out of
            // one `GuestRamImport`, which bounded them together, so the sum
            // cannot leave its space here.
            gpa_base: region
                .gpa_base
                .map(|base| base + (aligned_base - region.host_va)),
            host_va: region.host_va,
            len: region.len,
            page_size,
            aligned_base,
            aligned_len,
        });
    }
    Ok(candidates)
}

/// Take the whole guest-RAM import now, so the guest's first draw does not pay
/// for it.
///
/// # Two steps, and only the second one costs
///
/// Asking the host where its RAMBlocks are ([`resolve`]) is a handful of shim
/// calls. Handing each of those mappings to the GPU is `vkAllocateMemory` with
/// a host pointer chained, which is where a driver that pins takes a reference
/// on every page of guest RAM — seconds, proportional to the RAM the VM was
/// given, and measured per block by
/// [`crate::backend::vulkan::engine::warm_guest_ram_imports`].
///
/// Both were lazy and both landed on the guest's first `gather`, inside its
/// first draw, inside a display transaction the guest abandons after 1000 ms.
/// Moving only the first one bought nothing measurable, which is the finding
/// that located the second: `guest_ram_span` moved a second earlier and
/// `gather_us` did not move at all. Called from the guest driver's
/// protocol-version handshake, both now run before the guest has a display pipe
/// to arm a watchdog on, and every later caller finds the answer already there.
///
/// **It must never cache a negative.** [`resolve`] answers `NoBackendImport`
/// when no backend has published a granularity yet, and that answer is latched
/// in `MAP` for the rest of the boot — so warming before the backend is up
/// would turn a capable host into one that refuses every window, which is the
/// opposite of the intent and would look like a host that lacks the extension.
/// The guard is the same question `resolve` asks first, and asking it here
/// leaves the lazy path to handle a backend that is genuinely late.
///
/// The device-side half is Vulkan-only because only Vulkan has a device-side
/// half: the Metal-direct arm builds a `newBufferWithBytesNoCopy` per call
/// against unified memory and holds no per-RAMBlock import to warm.
pub fn warm<H: HostOps + ?Sized>(host: &mut H) {
    if granularity().is_none() {
        return;
    }
    let already = MAP.lock().unwrap_or_else(|p| p.into_inner()).is_some();
    if !already {
        with_map(host, |_| ());
    }
    let imports = imports();
    if !imports.is_empty() {
        let (warmed, bytes) = crate::backend::selected().warm_guest_ram_imports(&imports);
        if warmed > 0 {
            crate::observe::off(format!(
                "guest_ram_warm blocks={warmed} bytes={bytes} spans={}",
                imports.len()
            ));
        }
    }
}

/// Resolve on the first call of a boot, then run `body` against the result.
///
/// The one place the resolution is built, so no entry point can hold a second
/// copy of "have we asked the host yet".
fn with_map<H: HostOps + ?Sized, R>(host: &mut H, body: impl FnOnce(&Resolved) -> R) -> R {
    let mut guard = MAP.lock().unwrap_or_else(|p| p.into_inner());
    if guard.is_none() {
        *guard = Some(resolve(host));
    }
    body(guard.as_ref().expect("just resolved"))
}

/// The refusal the whole rail is standing on, if there is one.
///
/// An entry point that judges the *shape* of what it was given must ask this
/// first. The order is not cosmetic: on a host with no import every window
/// refuses, and one told it was too fragmented sends a reader hunting for a
/// contiguity problem that a contiguous window would not have fixed either —
/// which is exactly what a driven `REIMS_VGPU_GUEST_IMPORT=off` boot logged
/// before [`reference_for_pages`] asked.
///
/// Public because it is also the cheap early-out: a rail whose next step is an
/// `O(pages)` walk should ask this before paying for one it is going to throw
/// away. That caller must ask *this* rather than re-reading
/// [`crate::runtime::guest_ram::granularity`], which is the same answer for one
/// of the four refusals and silence for the other three.
pub fn standing_refusal<H: HostOps + ?Sized>(host: &mut H) -> Option<MapRefusal> {
    with_map(host, |resolved| resolved.refusal)
}

/// The whole admission rule for a packed-alias import, in one place.
///
/// The alias rails do something the RAMBlock map does not: they build a *new*
/// host-pointer import over a host allocation they arrange themselves, so that a
/// guest resource whose pages are scattered can still be presented to Vulkan as
/// one contiguous range. They are the only importers outside [`resolve`].
///
/// # Why they may not assemble this themselves
///
/// They did, and it was the same three latches written by hand twice —
/// `granularity`, `import_span_max`, `import_budget` — with the term that
/// matters missing from both copies: [`standing_refusal`]. That is precisely
/// what that function's doc forbids, in the sentence saying a caller must ask it
/// "rather than re-reading `guest_ram::granularity`, which is the same answer
/// for one of the four refusals and silence for the other three".
///
/// The silence is the bug. The latches are published once at device create from
/// what the *host* supports, and they stay `Some` for the whole boot. The map's
/// refusal is a different judgement made later, about what this *guest* fits
/// into that host — `ImportExceedsHeap` above all, where the guest's RAMBlocks
/// sum past the roomiest heap an import can be charged to. On such a host the
/// map is discarded and every RAMBlock reference declines, and these two rails
/// went on importing anyway, because every latch they asked still answered
/// `Some`.
///
/// That is the reported failure and not a theoretical one: an 8 GiB-or-larger
/// guest on a host whose importable heap is smaller boots to a black screen or
/// dies, while the same guest with `REIMS_VGPU_GUEST_IMPORT=off` works — and
/// `off` is the one arm that also takes these rails' latches away, which is why
/// it was the arm that worked.
///
/// So the rule is one function and the answer is the alignment to import at.
/// `None` is a refusal already named on the fail channel by whoever owns it.
pub fn packed_alias_import_align<H: HostOps + ?Sized>(host: &mut H, len: u64) -> Option<u64> {
    if let Some(refusal) = standing_refusal(host) {
        // Latched by the refusal, not per call: on a host standing on one,
        // every alias attempt of every resource refuses for the same reason and
        // a line each would drown the channel.
        crate::observe::Emit::decline("packed_alias_refused", &refusal)
            .fail_once(crate::observe::Decline::slug(&refusal).as_ptr() as u64);
        crate::runtime::drain::note_store_route("zc_packed_alias_standing_refusal");
        return None;
    }
    let align = crate::runtime::guest_ram::granularity()?;
    if len > crate::runtime::guest_ram::import_span_max()?
        || len > crate::runtime::guest_ram::import_budget()?
    {
        return None;
    }
    Some(align)
}

/// Turn a guest physical address and a length into a bindable reference.
///
/// The whole guest-memory rail goes through here. Building the imports on the
/// first call is why `host` is taken: after that it is not touched.
pub fn reference<H: HostOps + ?Sized>(
    host: &mut H,
    gpa: u64,
    len: u64,
) -> Result<GuestRef, MapRefusal> {
    with_map(host, |resolved| {
        if let Some(refusal) = resolved.refusal {
            return Err(report_once(refusal));
        }
        resolved.reference(gpa, len)
    })
}

/// One stretch of a scattered window: where it starts in the window, and the
/// import reference covering it.
///
/// `window_offset` is a byte offset from the first byte the caller asked for,
/// not from the start of a page and not from the start of the import. It is
/// what a copy's source offset is measured in, which is the only thing a
/// consumer needs and the only thing it may not compute for itself.
#[derive(Debug)]
pub struct GuestWindowRun {
    /// Byte offset of this run's first byte within the requested window.
    pub window_offset: u64,
    /// The bindable reference for this run's bytes.
    pub guest: GuestRef,
}

/// [`reference_for_pages`] for a window that is *not* one contiguous stretch:
/// one reference per maximal GPA run, in window order.
///
/// # Why this exists as well as [`reference_for_pages`]
///
/// A driven x86 boot measured every guest surface at four 4 KiB pages per run —
/// the guest backs a surface in 16 KiB physically-contiguous granules — so a
/// 1920x1080 window is 2025 pages in 507 runs and *always* will be. See
/// [`MapRefusal::Scattered`] for the distribution. A rail that only takes one
/// contiguous stretch therefore never runs on a real workload, which is what
/// the boot found: the import was `supported` and bound 8 KiB in 25 seconds.
///
/// # What the caller owes
///
/// Every returned run is a separate bind. A consumer that issues one GPU copy
/// per run is correct; one that concatenates them is not, because nothing
/// relates two runs' import offsets. The runs tile the window exactly — no
/// gaps, no overlaps, ascending — and the tests below assert that rather than
/// leaving it to be re-derived.
///
/// Runs are **not** bounded here. A bound belongs where the cost is, which is
/// the consumer's region array, and a cap in this function would silently hand
/// back a partial window — the failure mode that loses guest work quietly. A
/// consumer that cannot issue N copies must refuse by name on the count it got.
///
/// # One lock for the whole window
///
/// The resolution is behind a mutex, and this runs on the draw-time buffer rail
/// at ~16 000 windows a second of ~16 runs each. Resolving each run through
/// [`reference`] would take and drop that mutex a quarter of a million times a
/// second for an answer that cannot change inside one call, so the walk happens
/// inside a single [`with_map`] instead. [`reference`] keeps its own lock for
/// the callers that resolve exactly one span.
pub fn references_for_runs<H: HostOps + ?Sized>(
    host: &mut H,
    gpas: &[u64],
    page_size: u64,
    in_page: u64,
    len: u64,
) -> Result<Vec<GuestWindowRun>, MapRefusal> {
    if gpas.is_empty() || page_size == 0 || len == 0 {
        return Err(report_once(MapRefusal::Scattered {
            pages: gpas.len(),
            runs: 0,
            first: gpas.first().copied().unwrap_or(0),
        }));
    }
    // Absolute byte range this window occupies, measured from the first byte of
    // `gpas[0]` — the same frame `in_page` is stated in. Page indices and run
    // boundaries are both in this frame, so no step below re-derives it.
    let window_start = in_page;
    let window_end = in_page.checked_add(len).ok_or(MapRefusal::Scattered {
        pages: gpas.len(),
        runs: 0,
        first: gpas[0],
    })?;

    with_map(host, |resolved| {
        if let Some(refusal) = resolved.refusal {
            return Err(report_once(refusal));
        }
        let mut out = Vec::new();
        for run in reims_vgpu_paging::runs::contig_page_runs(gpas, page_size) {
            let run_start = (run.start as u64) * page_size;
            let run_end = (run.end as u64) * page_size;
            // Clip to the window: the first run usually starts before it (the
            // window begins `in_page` bytes in) and the last usually ends after
            // it.
            let start = run_start.max(window_start);
            let end = run_end.min(window_end);
            if start >= end {
                continue;
            }
            // GPA-contiguous is not import-contiguous. A RAMBlock is imported in
            // chunks, so a stretch the guest laid out as one run can cross a
            // seam between two of them — and a `GuestRef` is an offset into one
            // import, so it cannot describe both sides.
            //
            // Split at the seam rather than refuse at it. The consumers already
            // take a list and a RAMBlock boundary has always been able to
            // produce one, so two runs here are indistinguishable from two the
            // guest's own page plan produced. Refusing instead would drop the
            // *whole window* to the copying rail — a named, safe decline, but
            // one that fires on roughly one writeback in 250 for no reason the
            // guest could see, which is a chunk size leaking into throughput.
            //
            // Within a run the GPAs are contiguous by construction, so one add
            // reaches any byte of it.
            let mut piece = start;
            while piece < end {
                let gpa = gpas[run.start] + (piece - run_start);
                // `None` means nothing backs this address at all: hand the whole
                // remainder to `reference` so it names that refusal rather than
                // this loop inventing a second one.
                let piece_end = match resolved.import_end(gpa) {
                    Some(import_end) if import_end > gpa => end.min(piece + (import_end - gpa)),
                    _ => end,
                };
                out.push(GuestWindowRun {
                    window_offset: piece - window_start,
                    guest: resolved.reference(gpa, piece_end - piece)?,
                });
                piece = piece_end;
            }
        }
        if out.is_empty() {
            return Err(report_once(MapRefusal::Scattered {
                pages: gpas.len(),
                runs: 0,
                first: gpas[0],
            }));
        }
        Ok(out)
    })
}

/// [`reference`] for a decoded page list: `len` bytes starting `in_page` bytes
/// into `gpas[0]`.
///
/// The one implementation of the contiguity rule, so the sampled, buffer and
/// writeback rails cannot disagree about what a bindable page list is.
pub fn reference_for_pages<H: HostOps + ?Sized>(
    host: &mut H,
    gpas: &[u64],
    page_size: u64,
    in_page: u64,
    len: u64,
) -> Result<GuestRef, MapRefusal> {
    if let Some(refusal) = standing_refusal(host) {
        return Err(report_once(refusal));
    }
    let Some(&first) = gpas.first() else {
        return Err(report_once(MapRefusal::Scattered {
            pages: 0,
            runs: 0,
            first: 0,
        }));
    };
    let contiguous = gpas
        .iter()
        .enumerate()
        .all(|(i, gpa)| *gpa == first + (i as u64) * page_size);
    if !contiguous {
        return Err(report_once(MapRefusal::Scattered {
            pages: gpas.len(),
            runs: reims_vgpu_paging::runs::contig_run_count(gpas, page_size),
            first,
        }));
    }
    reference(host, first + in_page, len)
}

/// Ask the host where guest RAM lives and bound every span to the backend's
/// granularity.
///
/// # This used to run on the guest's first draw, and it is NOT the two seconds
///
/// It was lazy — the first `reference_for_pages` triggered it — and it now runs
/// at the guest driver's protocol handshake instead ([`warm`]). **That move was
/// measured and it did not shift the stall**, which is what located the real
/// one: asking the host where its RAM is costs nothing, and handing those
/// mappings to the GPU costs everything. The evidence is one timestamp —
/// `guest_ram_span`, emitted once per boot by this function, moved from t=56453
/// to t=55342 while `gather_us` on the first frame stayed at 2 180 583 over the
/// same six gathers.
///
/// The seconds are `vkAllocateMemory` with the host pointer chained, measured
/// per RAMBlock at [`crate::backend::vulkan::engine::warm_guest_ram_imports`],
/// which is now also warmed from [`warm`]. The table below is the state before
/// that, kept because its second row is what ruled out a per-byte cost and sent
/// the search to one-time setup:
///
/// ```text
///                 draw_stall     stage_us     gather_us  gather_n  gather_b
/// macos-11 first   2 028 844    2 022 252     2 022 259         6   1 176 768
/// macos-11 later           —            —            75        61  13 545 376
/// macos-13 first   1 959 875    1 951 567     1 951 562         4     523 904
/// macos-13 later           —            —           105       104  15 318 048
/// ```
///
/// Six gathers of 1.1 MB taking two seconds and sixty-one gathers of 13 MB
/// taking 75 µs is not a gather cost; it is one-time setup charged to whoever
/// arrives first. The same boots report it as a `sync_exec_lock_hold` of
/// ~2 000 000 µs over one to three draws.
///
/// **The guest has a one-second watchdog behind this.** Its display pipe waits
/// on a submitted display transaction and gives up after 1000 ms, so a first
/// frame that takes two seconds blows it on every boot of every rail. Both
/// rails measured here do blow it; the macos-13 guest recovers and the macos-11
/// guest does not, and on macos-11 that is the whole visible failure — the
/// transaction stays pending, WindowServer stops answering, and the session
/// never starts.
///
/// The same driven boot then timed the two halves of the import separately and
/// read `probe_us=0` beside `alloc_us=2 493 029` for a 15 032 385 536-byte
/// RAMBlock and `alloc_us=309 796` for a 2 146 435 072-byte one — the whole
/// stall, in the one call the first gather was the first to reach. That is what
/// [`warm`] now takes at the handshake.
///
/// Timings above are wall clock on a shared host and are upper bounds; the
/// counts and byte totals are not.
/// Split one RAMBlock into consecutive regions of at most `span_max` bytes.
///
/// The regions tile the block exactly and in ascending order: no byte is
/// dropped, none is covered twice, and the last one is whatever remains. A block
/// already inside the ceiling comes back as itself, so a host that needs no
/// chunking pays one comparison and allocates the same one-element shape it
/// always had.
///
/// Alignment is deliberately *not* applied here. `GuestRamImport::new` trims
/// each region to the device's granularity and names its own refusal when a
/// region cannot survive that — doing it twice would be two spellings of one
/// rule, and this one has no granularity in hand. The consequence is that
/// `span_max` must itself be a multiple of the granularity, which is why the
/// backend masks it before publishing and why a ceiling below the granularity is
/// refused outright rather than clamped.
fn chunk_span(span: GuestRamRegion, span_max: u64) -> Vec<GuestRamRegion> {
    if span_max == 0 || span.len <= span_max {
        return vec![span];
    }
    let mut out = Vec::with_capacity((span.len / span_max) as usize + 1);
    let mut done = 0u64;
    while done < span.len {
        let len = span_max.min(span.len - done);
        out.push(GuestRamRegion {
            host_va: span.host_va + done,
            gpa_base: span.gpa_base + done,
            len,
        });
        done += len;
    }
    out
}

fn resolve<H: HostOps + ?Sized>(host: &mut H) -> Resolved {
    let Some(align) = granularity() else {
        return Resolved {
            imports: Vec::new(),
            refusal: Some(MapRefusal::NoBackendImport),
        };
    };
    let spans = match host.guest_ram_regions() {
        Ok(spans) => spans,
        Err(why) => {
            return Resolved {
                imports: Vec::new(),
                refusal: Some(MapRefusal::HostRefused(why)),
            }
        }
    };
    let count = spans.len();
    // A span this device cannot bound is skipped rather than fatal: a machine
    // with one ordinary RAMBlock and one odd sliver should import the RAMBlock.
    // `GuestRamImport::new` names the check that rejected each skipped one on
    // the fail channel, so a partial import is never silent.
    //
    // One import per RAMBlock was the shape until a driver was found truncating
    // `allocationSize` to 32 bits on this path — see
    // [`crate::backend::vulkan::caps::host_pointer::IMPORT_SPAN_CEILING`]. Each
    // block is now imported in chunks no larger than the span the backend
    // published. Nothing else had to change: a window already resolves against
    // whichever import backs its GPA, and one straddling two of them already
    // groups into two `VkBuffer` sources, because a RAMBlock boundary has always
    // been able to split one.
    let span_max = import_span_max().unwrap_or(u64::MAX);
    let mut imports: Vec<Arc<GuestRamImport>> = spans
        .into_iter()
        .flat_map(|span| chunk_span(span, span_max))
        .filter_map(|span| GuestRamImport::new(span, align).ok().map(Arc::new))
        .collect();
    // `reference` binary-searches this, so the order is load-bearing rather than
    // cosmetic. The shim reports blocks in ascending GPA and `chunk_span` keeps
    // that, so this is a no-op on every machine seen so far — it is here because
    // the search would silently answer `GpaNotInAnyImport` for a live address if
    // a future shim ever reported them out of order.
    imports.sort_by_key(|i| i.gpa_base());
    // Every import is live for the VM's lifetime and any submission may name any
    // of them, so what has to fit is the sum and not the largest block. A guest
    // that does not fit takes the copying rails whole rather than in part — see
    // [`MapRefusal::ImportExceedsHeap`] for why a partial import is the one
    // outcome that is worse than either.
    let needed: u64 = imports.iter().map(|i| i.len()).sum();
    let over_budget = import_budget().filter(|budget| needed > *budget);
    if let Some(budget) = over_budget {
        return Resolved {
            imports: Vec::new(),
            refusal: Some(MapRefusal::ImportExceedsHeap { needed, budget }),
        };
    }
    let refusal = imports
        .is_empty()
        .then_some(MapRefusal::NoUsableRegion { spans: count });
    // Once per boot, because this is what makes `guest_import_levels`'s
    // denominator interpretable. That line reports `imported/reported` and a
    // reader seeing `1/4` cannot tell which three went untouched, or whether the
    // untouched ones are guest RAM at all — on q35 the reported set is the two
    // halves of `-m` either side of the PCI hole plus whatever smaller writable
    // RAM regions the board exposes. Naming each span's base and length answers
    // that from the log instead of from a comment that would go stale when a
    // board changes.
    //
    // `resolve` runs once per boot (and again only after a device teardown), so
    // this is a handful of lines, not a cadence.
    for (n, import) in imports.iter().enumerate() {
        crate::observe::off(format!(
            "guest_ram_span n={n}/{count} gpa={:#x} len={} mib={}",
            import.gpa_base().expect("RAMBlock imports have a GPA base"),
            import.len(),
            import.len() / (1024 * 1024),
        ));
    }
    Resolved { imports, refusal }
}

/// Emit `refusal` and hand it back.
///
/// Deduped by slug: these are per-reference and a decode path that names an
/// unbacked address once will name it every frame.
/// [`MapRefusal::NoBackendImport`] goes to the off channel — it is the host
/// saying what it is, not a loss of guest work — and everything else to the
/// fail channel.
fn report_once(refusal: MapRefusal) -> MapRefusal {
    let line = crate::observe::Emit::decline(EVENT, &refusal);
    match refusal {
        MapRefusal::NoBackendImport => {
            if crate::observe::first_sight("guest_ram_map_no_backend_import", 0) {
                line.off();
            }
        }
        _ => line.fail_once(0),
    }
    refusal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::guest_ram::{forget_import_limits, latch_import_limits, GuestRamRegion};

    /// A budget no test span can exceed, so a test about the granularity is not
    /// also a test about the heap. The budget tests name their own.
    const UNBOUNDED: u64 = u64::MAX;

    /// Latch a granularity with a budget and a span ceiling that admit
    /// everything, so a test about the granularity is only about that.
    fn latch_granularity(align: u64) {
        latch_import_limits(align, UNBOUNDED, UNBOUNDED);
    }

    fn forget_granularity() {
        forget_import_limits();
    }

    /// The whole module is process-global, and so is the granularity latch.
    /// Every test here takes this and restores both.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct Spans(Vec<GuestRamRegion>);

    impl HostOps for Spans {
        fn mono_ns(&self) -> u64 {
            0
        }
        fn enqueue(&mut self, _action: crate::runtime::host::HostAction) {}
        fn schedule_bh(&mut self) {}
        fn guest_ram_regions(&mut self) -> Result<Vec<GuestRamRegion>, GuestRamRegionsError> {
            Ok(self.0.clone())
        }
    }

    struct Refusing;

    impl HostOps for Refusing {
        fn mono_ns(&self) -> u64 {
            0
        }
        fn enqueue(&mut self, _action: crate::runtime::host::HostAction) {}
        fn schedule_bh(&mut self) {}
        fn guest_ram_regions(&mut self) -> Result<Vec<GuestRamRegion>, GuestRamRegionsError> {
            Err(GuestRamRegionsError::NoRam)
        }
    }

    /// Two spans with a hole between them, which is the shape of an x86 machine
    /// with a PCI hole and the reason the lookup is a search rather than a
    /// single subtraction.
    fn two_spans() -> Spans {
        Spans(vec![
            GuestRamRegion {
                gpa_base: 0,
                host_va: 0x7f00_0000_0000,
                len: 0x8000_0000,
            },
            GuestRamRegion {
                gpa_base: 0x1_0000_0000,
                host_va: 0x7f80_0000_0000,
                len: 0x8000_0000,
            },
        ])
    }

    fn with_granularity<R>(align: Option<u64>, body: impl FnOnce() -> R) -> R {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset();
        match align {
            Some(a) => latch_granularity(a),
            None => forget_granularity(),
        }
        let out = body();
        reset();
        forget_granularity();
        out
    }

    /// Run `body` with a granularity and an explicit span ceiling latched, so a
    /// test about chunking is not also a test about the heap.
    fn with_span_max<R>(align: u64, span_max: u64, body: impl FnOnce() -> R) -> R {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset();
        latch_import_limits(align, UNBOUNDED, span_max);
        let out = body();
        reset();
        forget_import_limits();
        out
    }

    /// The chunks must tile the block: every byte covered once, in order, none
    /// invented and none dropped.
    ///
    /// Written as a walk rather than as an expected list because the failure
    /// this guards is an off-by-one at a boundary, and a hand-written list of
    /// expected chunks is exactly as likely to carry the same off-by-one as the
    /// code it checks. The three sizes are the three shapes: an exact multiple, a
    /// remainder, and a block already inside the ceiling.
    #[test]
    fn chunking_tiles_the_block_exactly() {
        for (len, span_max) in [
            (0x8000u64, 0x2000u64), // exact multiple
            (0x9001, 0x2000),       // ragged remainder
            (0x1000, 0x2000),       // already inside the ceiling
            (0x2000, 0x2000),       // exactly the ceiling
        ] {
            let span = GuestRamRegion {
                gpa_base: 0x1_0000_0000,
                host_va: 0x7f00_0000_0000,
                len,
            };
            let chunks = chunk_span(span, span_max);
            assert!(!chunks.is_empty(), "len={len:#x} produced no chunk");
            let mut walked = 0u64;
            for c in &chunks {
                assert!(c.len > 0, "len={len:#x} produced an empty chunk");
                assert!(
                    c.len <= span_max,
                    "len={len:#x} chunk of {:#x} is past the ceiling {span_max:#x}",
                    c.len
                );
                assert_eq!(
                    c.gpa_base,
                    span.gpa_base + walked,
                    "len={len:#x} chunk does not continue where the last ended"
                );
                assert_eq!(
                    c.host_va,
                    span.host_va + walked,
                    "len={len:#x} host pointer drifted from the GPA"
                );
                walked += c.len;
            }
            assert_eq!(walked, len, "len={len:#x} chunks do not cover the block");
        }
    }

    /// The reason chunking exists: no single import may be longer than the span
    /// the backend published.
    ///
    /// A driver has been measured truncating `allocationSize` to 32 bits on the
    /// host-pointer path and reading back unrelated data past the truncation, so
    /// an import longer than the published ceiling is not a slow import, it is a
    /// silently wrong one. This asserts the property that makes that
    /// unreachable, and asserts it about every import rather than about the
    /// count, because the count is a consequence and the length is the rule.
    #[test]
    fn no_import_is_longer_than_the_span_the_backend_published() {
        const CEILING: u64 = 0x2000_0000;
        with_span_max(0x1000, CEILING, || {
            let mut host = two_spans();
            assert_eq!(standing_refusal(&mut host), None);
            let imports = imports();
            assert_eq!(
                imports.len(),
                8,
                "two 2 GiB blocks at a 512 MiB ceiling are four chunks each"
            );
            for i in &imports {
                assert!(
                    i.len() <= CEILING,
                    "import at {:#x} is {} bytes, past the {CEILING} ceiling",
                    i.gpa_base().expect("RAMBlock imports have a GPA base"),
                    i.len()
                );
            }
            assert_eq!(
                imports.iter().map(|i| i.len()).sum::<u64>(),
                two_spans_bytes(),
                "chunking must not lose or duplicate guest RAM"
            );
        });
    }

    /// Every address the unchunked map resolved must still resolve, including
    /// the ones that now fall in a later chunk — and a hole must still be
    /// refused, which is what says the search did not simply widen.
    ///
    /// The boundary addresses are the point. `partition_point` picks the last
    /// import whose base is `<= gpa`, so a GPA exactly on a chunk base is the
    /// case an off-by-one puts in the previous chunk, where it is one byte past
    /// the end.
    #[test]
    fn a_chunked_block_resolves_at_every_boundary_and_still_refuses_the_hole() {
        const CEILING: u64 = 0x2000_0000;
        with_span_max(0x1000, CEILING, || {
            let mut host = two_spans();
            for gpa in [
                0u64,
                CEILING - 0x1000,
                CEILING,
                CEILING + 0x1000,
                0x8000_0000 - 0x1000,
                0x1_0000_0000,
                0x1_0000_0000 + CEILING,
                0x1_8000_0000 - 0x1000,
            ] {
                let got = reference(&mut host, gpa, 0x1000);
                assert!(
                    got.is_ok(),
                    "gpa {gpa:#x} is guest RAM and did not resolve: {:?}",
                    got.err()
                );
            }
            assert_eq!(
                reference(&mut host, 0x8000_0000, 0x1000).err(),
                Some(MapRefusal::GpaNotInAnyImport { gpa: 0x8000_0000 }),
                "the PCI hole is not guest RAM and chunking must not have covered it"
            );
        });
    }

    /// A guest run that crosses a seam between two chunks of one RAMBlock is
    /// **split**, not refused, and the pieces tile the request exactly.
    ///
    /// A `GuestRef` is an offset into one import, so it cannot describe both
    /// sides of a seam. Refusing would be safe — a named decline, whole window
    /// to the copying rail — and would put a chunk size the guest cannot see
    /// into this device's throughput on roughly one writeback in 250. The
    /// consumers already take a list, and a RAMBlock boundary has always been
    /// able to produce one, so a split here is indistinguishable from a split
    /// the guest's own page plan produced.
    ///
    /// Asserted on the tiling rather than on the count, because "two runs" would
    /// still pass if the second one started in the wrong place.
    #[test]
    fn a_run_crossing_a_chunk_seam_is_split_and_the_pieces_tile_the_window() {
        const CEILING: u64 = 0x2000_0000;
        const PAGE: u64 = 0x1000;
        with_span_max(PAGE, CEILING, || {
            let mut host = two_spans();
            // Two GPA-contiguous pages either side of the first chunk seam.
            let gpas = [CEILING - PAGE, CEILING];
            let runs = references_for_runs(&mut host, &gpas, PAGE, 0, 2 * PAGE)
                .expect("both pages are guest RAM; a seam is not a refusal");

            assert_eq!(runs.len(), 2, "one piece per import the run touches");
            let mut want = 0u64;
            for r in &runs {
                assert_eq!(
                    r.window_offset, want,
                    "a piece does not begin where the last one ended"
                );
                want += r.guest.requested();
            }
            assert_eq!(want, 2 * PAGE, "the pieces do not cover the window");
            assert!(
                runs[0].guest.import().id() != runs[1].guest.import().id(),
                "a split that stays inside one import is not a seam split"
            );
        });
    }

    /// The same run, with the seam moved out of its way, must stay one piece.
    /// Without this the test above would pass on an implementation that split
    /// every run in half.
    #[test]
    fn a_run_inside_one_chunk_is_not_split() {
        const CEILING: u64 = 0x2000_0000;
        const PAGE: u64 = 0x1000;
        with_span_max(PAGE, CEILING, || {
            let mut host = two_spans();
            let gpas = [CEILING + PAGE, CEILING + 2 * PAGE];
            let runs = references_for_runs(&mut host, &gpas, PAGE, 0, 2 * PAGE)
                .expect("both pages are guest RAM");
            assert_eq!(runs.len(), 1, "a run wholly inside one import is one piece");
            assert_eq!(runs[0].guest.requested(), 2 * PAGE);
        });
    }

    /// Run `body` with a granularity and an explicit heap budget latched.
    fn with_budget<R>(align: u64, budget: u64, body: impl FnOnce() -> R) -> R {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset();
        latch_import_limits(align, budget, UNBOUNDED);
        let out = body();
        reset();
        forget_import_limits();
        out
    }

    /// The bytes [`two_spans`] asks this device to keep resident. Derived from
    /// the spans rather than written out, so a change to either stays one number.
    fn two_spans_bytes() -> u64 {
        two_spans().0.iter().map(|r| r.len).sum()
    }

    /// A guest larger than the roomiest heap on the host's GPU must not import
    /// at all.
    ///
    /// This is the reported `radv`/`amdgpu` failure: the import succeeds, and
    /// every submission that names it then fails validation in the kernel with
    /// `Not enough memory for command submission`, which arrives as a lost
    /// device. The copying rails are the working configuration on such a host and
    /// this is how the device reaches them without an operator setting a variable.
    #[test]
    fn a_guest_larger_than_every_heap_takes_the_copying_rails() {
        let needed = two_spans_bytes();
        with_budget(0x1000, needed - 1, || {
            let mut host = two_spans();
            assert_eq!(
                standing_refusal(&mut host),
                Some(MapRefusal::ImportExceedsHeap {
                    needed,
                    budget: needed - 1,
                }),
                "a guest past the budget must refuse by name"
            );
            assert_eq!(
                imports().len(),
                0,
                "and must not leave a partial import behind"
            );
        });
    }

    /// The bound is `>`, not `>=`: a guest that exactly fills the roomiest heap
    /// is admitted, because nothing in the contract says an allocation the size
    /// of its heap cannot be made. Pinned because an off-by-one in the safe
    /// direction here silently costs every host whose heap equals its guest.
    #[test]
    fn a_guest_that_exactly_fills_the_roomiest_heap_still_imports() {
        let needed = two_spans_bytes();
        with_budget(0x1000, needed, || {
            let mut host = two_spans();
            assert_eq!(standing_refusal(&mut host), None);
            assert_eq!(imports().len(), 2);
        });
    }

    /// A backend that publishes a granularity beside a zero budget has published
    /// nothing: the pair is withdrawn, and the map reads it as a host that cannot
    /// import. Without this, a zero budget would refuse every guest by name and
    /// blame the guest's size for a backend that answered badly.
    #[test]
    fn a_zero_budget_withdraws_the_granularity_rather_than_refusing_every_guest() {
        with_budget(0x1000, 0, || {
            let mut host = two_spans();
            assert_eq!(
                standing_refusal(&mut host),
                Some(MapRefusal::NoBackendImport),
                "a zero budget is a backend that cannot import, not a guest that is too big"
            );
        });
    }

    /// Warming before a backend has published a granularity must leave the
    /// resolution untaken, so the import a late backend enables is still
    /// available.
    ///
    /// This is the whole hazard in moving the import off the guest's first
    /// draw. `resolve` answers `NoBackendImport` when there is no granularity
    /// and that answer is latched in `MAP` for the rest of the boot, so a warm
    /// taken one instant too early does not merely fail to help — it converts a
    /// host that can import into one that refuses every window for the life of
    /// the VM, and reports it as a host lacking the extension.
    #[test]
    fn warming_before_the_backend_publishes_a_granularity_latches_nothing() {
        with_granularity(None, || {
            let mut host = two_spans();
            warm(&mut host);
            assert!(
                MAP.lock().unwrap_or_else(|p| p.into_inner()).is_none(),
                "a warm with no granularity must not latch a refusal"
            );

            // The backend comes up late; the import must still be available.
            latch_granularity(0x1000);
            warm(&mut host);
            assert_eq!(
                standing_refusal(&mut host),
                None,
                "the late backend's import must still resolve"
            );
            assert_eq!(imports().len(), 2, "both spans import once warmed");
        });
    }

    /// Warming is what the first reference would have done, and doing it twice
    /// changes nothing — the guest's handshake may be replayed, and every later
    /// caller has to find the same answer.
    #[test]
    fn warming_is_idempotent_and_equals_what_the_lazy_path_would_resolve() {
        let lazy = with_granularity(Some(0x1000), || {
            let mut host = two_spans();
            let refusal = standing_refusal(&mut host);
            (refusal, imports().len())
        });
        let warmed = with_granularity(Some(0x1000), || {
            let mut host = two_spans();
            warm(&mut host);
            warm(&mut host);
            let refusal = standing_refusal(&mut host);
            (refusal, imports().len())
        });
        assert_eq!(warmed, lazy);
        assert_eq!(warmed.1, 2);
    }

    /// An address in either span resolves, and the offset it resolves to is the
    /// one inside *that* span — not a distance from the first one. Getting this
    /// wrong on a machine with a PCI hole binds the GPU 4 GiB away from the
    /// bytes the guest named, inside a live import, where no bound would catch
    /// it.
    #[test]
    fn an_address_resolves_against_the_span_that_backs_it() {
        with_granularity(Some(0x1000), || {
            let mut host = two_spans();
            let low = reference(&mut host, 0x2000, 0x100).expect("in the first span");
            assert_eq!(low.bound().expect("checked").offset, 0x2000);

            let high = reference(&mut host, 0x1_0000_2000, 0x100).expect("in the second span");
            assert_eq!(
                high.bound().expect("checked").offset,
                0x2000,
                "the offset is into the second import, not from the first span's base"
            );
            assert_ne!(low.import().id(), high.import().id());
        });
    }

    /// The span census counts every span the shim reported, not the ones a
    /// workload happened to touch.
    ///
    /// This is the denominator of the `guest_import_levels` census line, and it
    /// is the whole reason that line carries two terms. A backend imports a span
    /// at its first reference, so on this two-span machine — the shape q35 has
    /// with a PCI hole, which is what `vm/boot-x86.sh` boots — a workload that
    /// only ever touches high memory leaves the imported count at one. Reported
    /// against a denominator of one that would read as half of guest RAM having
    /// gone missing; against two it reads as lazy, which is what it is.
    #[test]
    fn the_span_census_counts_what_the_shim_reported_not_what_was_touched() {
        with_granularity(Some(0x1000), || {
            assert_eq!(
                span_census(),
                (0, 0),
                "nothing is asked before the first reference"
            );
            let mut host = two_spans();
            reference(&mut host, 0x1_0000_2000, 0x100).expect("in the second span");
            assert_eq!(
                span_census(),
                (2, 0x1_0000_0000),
                "both spans are reported though only the second was referenced"
            );
        });
    }

    /// The projection describes every import this process holds, field for
    /// field, inventing no byte and dropping none.
    ///
    /// Asserted against [`imports()`] rather than against a written-out list of
    /// expected regions. A list here would pin whatever this file happened to do
    /// the day it was written, and the chunk count is a consequence of a
    /// ceiling the test itself sets; what matters is that the two views of one
    /// map agree, that every region carries a distinct identity, and that the
    /// covered bytes are exactly the RAMBlocks the shim reported.
    #[test]
    fn the_projection_describes_every_import_this_process_holds() {
        const CEILING: u64 = 0x2000_0000;
        with_span_max(0x1000, CEILING, || {
            let mut host = two_spans();
            let projected = host_regions(&mut host).expect("a shim that answered can be projected");
            let imported = imports();

            assert_eq!(
                projected.len(),
                8,
                "two 2 GiB blocks at a 512 MiB ceiling are four chunks each"
            );
            assert_eq!(
                projected.len(),
                imported.len(),
                "the projection and the import list are two views of one map"
            );
            for (projected, import) in projected.iter().zip(imported.iter()) {
                assert_eq!(projected.import, import.id());
                assert_eq!(projected.gpa_base, import.gpa_base());
                assert_eq!(projected.host_va as usize, import.host_base());
                assert_eq!(projected.len, import.len());
                assert_eq!(
                    projected.page_size,
                    import.align(),
                    "the projection must name the granularity the import was built against"
                );
                assert_eq!(
                    projected.len % projected.page_size,
                    0,
                    "an import is a whole number of granules by construction"
                );
            }

            // The shape a reader uses this for: two blocks with the PCI hole
            // between them, in ascending order, each region a distinct identity.
            // The field comparison above would still pass against a projection
            // that collapsed the machine into one range.
            let bases: Vec<u64> = projected.iter().filter_map(|r| r.gpa_base).collect();
            assert_eq!(bases.first(), Some(&0), "the first RAMBlock is missing");
            assert!(
                bases.contains(&0x1_0000_0000),
                "the second RAMBlock is missing"
            );
            assert!(
                bases.windows(2).all(|w| w[0] < w[1]),
                "the projection is not in ascending GPA order: {bases:x?}"
            );
            let identities: std::collections::HashSet<_> =
                projected.iter().map(|r| r.import).collect();
            assert_eq!(
                identities.len(),
                projected.len(),
                "two regions share an import identity"
            );
            assert_eq!(
                projected.iter().map(|r| r.len).sum::<u64>(),
                two_spans_bytes(),
                "the projection must cover exactly the RAM the shim reported"
            );
        });
    }

    /// Nothing has been resolved yet, so nothing is projected — and the attempt
    /// latches no refusal on its way out.
    ///
    /// This is [`warm`]'s hazard in another spelling: a projection that resolved
    /// before the backend published a granularity would latch `NoBackendImport`
    /// in `MAP` for the rest of the boot, and the late backend's import would
    /// never appear. The second half is what makes this a test rather than a
    /// tautology — once the granularity arrives, the same call has to describe
    /// both spans.
    #[test]
    fn a_projection_before_the_backend_publishes_a_granularity_is_empty_and_latches_nothing() {
        with_granularity(None, || {
            let mut host = two_spans();
            assert!(
                host_regions(&mut host)
                    .expect("no callback was asked, so nothing can have refused")
                    .is_empty(),
                "nothing has been resolved, so there is nothing to project"
            );
            assert!(
                MAP.lock().unwrap_or_else(|p| p.into_inner()).is_none(),
                "a projection with no granularity must not latch a refusal"
            );

            latch_granularity(0x1000);
            assert_eq!(
                host_regions(&mut host)
                    .expect("the late backend's import resolves")
                    .len(),
                2,
                "the late backend's import must still be there to project"
            );
        });
    }

    /// A host that answers with nothing importable projects an empty list, and
    /// the emptiness is the host's rather than a refusal this dropped.
    ///
    /// The reason stays askable, which is the property that lets a caller treat
    /// "empty" as a fact about the device instead of inferring one.
    #[test]
    fn a_host_with_no_importable_span_projects_nothing() {
        with_granularity(Some(0x1000), || {
            let mut host = Spans(Vec::new());
            assert_eq!(
                host_regions(&mut host).expect("an empty answer is not a refusal"),
                Vec::new()
            );
            assert_eq!(
                standing_refusal(&mut host),
                Some(MapRefusal::NoUsableRegion { spans: 0 }),
                "the empty projection must be the host's answer, not a swallowed refusal"
            );
        });
    }

    /// A host that cannot say where guest RAM lives is an error and not an empty
    /// projection, and it keeps the name of its refusal.
    ///
    /// This is the difference the projection exists to preserve: "this shim
    /// cannot answer" sends a reader to a version mismatch, while "nothing is
    /// imported yet" is a device that has not been asked. A projection that
    /// folded the first into the second would read, on every fixture host and
    /// every pre-v17 shim, exactly like a device waiting for its first draw.
    #[test]
    fn a_host_without_the_callback_is_an_error_rather_than_an_empty_projection() {
        with_granularity(Some(0x1000), || {
            /// Implements everything but the one callback under test, so it
            /// takes `HostOps`'s own default for it — which is what a fixture
            /// host and a pre-v17 shim look like from here.
            struct NoCallback;
            impl HostOps for NoCallback {
                fn mono_ns(&self) -> u64 {
                    0
                }
                fn enqueue(&mut self, _action: crate::runtime::host::HostAction) {}
                fn schedule_bh(&mut self) {}
            }
            assert_eq!(
                host_regions(&mut NoCallback).err(),
                Some(GuestRamRegionsError::CallbackMissing),
                "'no callback' must not read as 'no imports'"
            );

            // `MAP` latched that refusal for the boot — which is what a device
            // teardown drops — and every other shim refusal has to keep its own
            // name rather than arriving as the one latched first.
            reset();
            assert_eq!(
                host_regions(&mut Refusing).err(),
                Some(GuestRamRegionsError::NoRam),
                "a machine with no RAM span is a different thing to go fix"
            );
        });
    }

    /// A live import identity to stamp on hand-built regions.
    ///
    /// `ImportId` is allocated inside the memory crate and cannot be forged
    /// here, so resolving a map is the only way to obtain a value of the type.
    /// Nothing in the bound reads the identity, which is why one resolution
    /// covers every hand-built shape below.
    fn a_live_import() -> ImportId {
        with_granularity(Some(0x1000), || {
            let mut host = two_spans();
            host_regions(&mut host)
                .expect("a shim that answered can be projected")
                .into_iter()
                .next()
                .expect("two spans resolve")
                .import
        })
    }

    /// A projected region with a GPA base over it, since every import this
    /// process holds today is a RAMBlock one.
    fn projected_region(import: ImportId, host_va: u64, len: u64, page_size: u64) -> HostRegion {
        HostRegion {
            import,
            gpa_base: Some(0x1_0000_0000),
            host_va,
            len,
            page_size,
        }
    }

    /// A region that is already a whole number of granules registers exactly as
    /// projected: the rounding is a no-op and not a shift in either direction.
    #[test]
    fn an_already_aligned_region_registers_unchanged() {
        let import = a_live_import();
        let projected = projected_region(import, 0x7f00_0000_0000, 0x2000, 0x1000);
        let candidates =
            registration_candidates(std::slice::from_ref(&projected)).expect("registerable");
        assert_eq!(
            candidates,
            vec![RegistrationCandidate {
                region_index: 0,
                import,
                gpa_base: Some(0x1_0000_0000),
                host_va: 0x7f00_0000_0000,
                len: 0x2000,
                page_size: 0x1000,
                aligned_base: 0x7f00_0000_0000,
                aligned_len: 0x2000,
            }],
            "an aligned region with a whole number of granules is carried through"
        );
    }

    /// Nothing projected is nothing to bound — and not an error.
    #[test]
    fn a_projection_with_no_regions_has_no_candidates() {
        assert_eq!(
            registration_candidates(&[]).expect("nothing to bound"),
            Vec::new()
        );
    }

    /// A base inside a granule rounds *forward*: the candidate never starts
    /// before the bytes it names.
    ///
    /// Both roundings are exercised at once here, because a range that starts
    /// mid-granule and ends mid-granule is the ordinary shape a caller passing
    /// the shim's raw `host_va` produces — and the shape whose two mistakes are
    /// a base rounded down (outside the region) and a length that keeps the
    /// partial first granule it just rounded past.
    #[test]
    fn a_base_inside_a_granule_rounds_forward_into_the_region() {
        let import = a_live_import();
        // Three and a half granules, starting 0x180 in: the base moves up 0xE80
        // to the next granule and the end comes back 0x980, leaving the two
        // whole granules between them.
        let projected = projected_region(import, 0x7f00_0000_0180, 0x3800, 0x1000);
        let candidates = registration_candidates(&[projected]).expect("registerable");
        assert_eq!(candidates.len(), 1);
        let candidate = candidates[0];
        assert_eq!(
            candidate.aligned_base, 0x7f00_0000_1000,
            "the base must round up, not down"
        );
        assert_eq!(candidate.aligned_len, 0x2000, "the end must round down");
        assert!(
            candidate.aligned_base >= candidate.host_va,
            "a candidate may not start before the region"
        );
        assert!(
            candidate.aligned_base + candidate.aligned_len <= candidate.host_va + candidate.len,
            "a candidate may not end past the region"
        );
        assert_eq!(
            candidate.gpa_base,
            Some(0x1_0000_0000 + 0xE80),
            "the GPA must move with the base, or the window would name the wrong guest bytes"
        );
        assert_eq!(
            candidate.len, 0x3800,
            "the projection is carried, not rewritten"
        );
    }

    /// A ragged end rounds back to the last whole granule, and the partial one
    /// is not covered.
    #[test]
    fn a_ragged_end_rounds_back_to_a_whole_granule() {
        let import = a_live_import();
        let projected = projected_region(import, 0x7f00_0000_0000, 0x2800, 0x1000);
        let candidates = registration_candidates(&[projected]).expect("registerable");
        assert_eq!(candidates.len(), 1);
        let candidate = candidates[0];
        assert_eq!(candidate.aligned_base, 0x7f00_0000_0000);
        assert_eq!(
            candidate.aligned_len, 0x2000,
            "the last partial granule is outside the candidate"
        );
        assert_eq!(
            candidate.gpa_base,
            Some(0x1_0000_0000),
            "an aligned base moves no GPA"
        );
    }

    /// A region with no whole granule left produces nothing, and the regions
    /// around it keep their own indices.
    ///
    /// Three shapes, each one granule or less of usable range: a region already
    /// aligned but shorter than a granule, a region whose aligned base lands
    /// exactly on its end, and an unaligned region whose tail is a partial
    /// granule that the base rounds past. The kept region is last so the index
    /// assertion can only pass if the index is the projection's own position
    /// rather than a running count over the survivors.
    #[test]
    fn a_region_with_no_whole_granule_left_produces_no_candidate() {
        let import = a_live_import();
        let shorter_than_a_granule = projected_region(import, 0x7f00_0000_0000, 0x800, 0x1000);
        let ends_on_the_boundary = projected_region(import, 0x7f00_0000_0800, 0x800, 0x1000);
        let kept = projected_region(import, 0x7f00_1000_0000, 0x1000, 0x1000);
        let candidates =
            registration_candidates(&[shorter_than_a_granule, ends_on_the_boundary, kept])
                .expect("a region with no granule is skipped, not refused");
        assert_eq!(
            candidates.len(),
            1,
            "a range shorter than a granule is not registerable and must not be invented"
        );
        assert_eq!(
            candidates[0].region_index, 2,
            "the index is the projection's position, not a count of the survivors"
        );
        assert_eq!(candidates[0].import, kept.import);
        assert_eq!(candidates[0].aligned_base, 0x7f00_1000_0000);
        assert_eq!(candidates[0].aligned_len, 0x1000);
    }

    /// A granularity no mask can express is refused, and the refusal names the
    /// region it came from.
    #[test]
    fn a_granularity_no_mask_can_express_is_refused() {
        let import = a_live_import();
        for page_size in [0, 0x600, 0x1001] {
            let projected = projected_region(import, 0x7f00_0000_0000, 0x2000, page_size);
            assert_eq!(
                registration_candidates(&[projected]).err(),
                Some(RegistrationRefusal::PageSizeNotPowerOfTwo {
                    region_index: 0,
                    page_size,
                }),
                "page_size {page_size:#x} must be refused, not clamped"
            );
        }
        // One byte is a power of two, so it is a legal granule: the check is
        // about the mask the rounding needs, not about a minimum page size. An
        // over-eager refusal here would take a host's import away for no reason.
        let one_byte_granule = projected_region(import, 0x7f00_0000_0000, 0x10, 1);
        assert_eq!(
            registration_candidates(&[one_byte_granule])
                .expect("one byte is a power of two")
                .len(),
            1
        );

        // The index travels with the refusal, so a log line points at the region
        // a reader can count to rather than at whichever one was first.
        let good = projected_region(import, 0x7f00_0000_0000, 0x1000, 0x1000);
        let bad = projected_region(import, 0x7f00_1000_0000, 0x1000, 3);
        let refusal = registration_candidates(&[good, bad]).err();
        assert_eq!(
            refusal,
            Some(RegistrationRefusal::PageSizeNotPowerOfTwo {
                region_index: 1,
                page_size: 3,
            })
        );
        assert_eq!(
            crate::observe::Decline::slug(&refusal.expect("refused")),
            "guest_ram_registration_page_size_not_power_of_two",
            "each check spells its own reason in the fail log"
        );
    }

    /// A region whose end leaves the address space is refused, while a base
    /// that merely cannot be rounded up is skipped.
    ///
    /// The two are different judgements and must stay different: the first is a
    /// malformed region with no end to bound, the second is a region whose last
    /// granule is incomplete — the same thing every other sub-granule region is.
    #[test]
    fn a_region_whose_end_leaves_the_address_space_is_refused() {
        let import = a_live_import();
        let host_va = u64::MAX - 0x10;
        let wraps = projected_region(import, host_va, 0x100, 0x1000);
        assert_eq!(
            registration_candidates(&[wraps]).err(),
            Some(RegistrationRefusal::RegionLeavesAddressSpace {
                region_index: 0,
                host_va,
                len: 0x100,
            })
        );
        assert_eq!(
            crate::observe::Decline::slug(&RegistrationRefusal::RegionLeavesAddressSpace {
                region_index: 0,
                host_va,
                len: 0x100,
            }),
            "guest_ram_registration_region_leaves_address_space"
        );

        // Ends exactly at the top of the address space, so nothing wraps — and
        // no aligned base exists, which leaves no whole granule behind.
        let flush_against_the_top = projected_region(import, u64::MAX - 0x10, 0x10, 0x1000);
        assert_eq!(
            registration_candidates(&[flush_against_the_top]).expect("nothing wrapped"),
            Vec::new()
        );
    }

    /// Over a stood-up map the candidates are the projection, bounded — and on a
    /// real projection both bounds are no-ops.
    ///
    /// This is the pairing the two functions exist for: [`host_regions`] says
    /// what this process holds and [`registration_candidates`] says what a
    /// registration may name, and on the imports this module actually builds the
    /// second answer is the first one unchanged. The equality assertions are the
    /// regression: a bound that started trimming a real import would mean the
    /// import was never trimmed at construction, which every other rail depends
    /// on.
    #[test]
    fn every_candidate_over_a_stood_up_map_is_the_projection_it_came_from() {
        with_granularity(Some(0x1000), || {
            let mut host = two_spans();
            let projected = host_regions(&mut host).expect("a shim that answered can be projected");
            let candidates =
                registration_candidates(&projected).expect("a projection is registerable");
            assert_eq!(
                candidates.len(),
                projected.len(),
                "every import is registerable, so none may be dropped"
            );
            for (index, (candidate, region)) in candidates.iter().zip(projected.iter()).enumerate()
            {
                assert_eq!(
                    candidate.region_index, index,
                    "the projection's own position"
                );
                assert_eq!(candidate.import, region.import);
                assert_eq!(candidate.page_size, region.page_size);
                assert_eq!(candidate.host_va, region.host_va);
                assert_eq!(candidate.len, region.len);
                assert_eq!(
                    candidate.aligned_base, region.host_va,
                    "an import is trimmed to its granularity when it is built"
                );
                assert_eq!(candidate.aligned_len, region.len);
                assert_eq!(
                    candidate.gpa_base, region.gpa_base,
                    "a bound that moves nothing must move no GPA"
                );
                assert_eq!(candidate.aligned_base % candidate.page_size, 0);
                assert_eq!(candidate.aligned_len % candidate.page_size, 0);
                assert!(candidate.aligned_len > 0, "a candidate covers a granule");
            }
        });
    }

    /// The imports are built once. The shim's answer does not change, and
    /// re-asking would be an address-space walk per guest reference.
    #[test]
    fn the_host_is_asked_once_however_many_references_follow() {
        with_granularity(Some(0x1000), || {
            struct Counting(std::cell::Cell<usize>);
            impl HostOps for Counting {
                fn mono_ns(&self) -> u64 {
                    0
                }
                fn enqueue(&mut self, _a: crate::runtime::host::HostAction) {}
                fn schedule_bh(&mut self) {}
                fn guest_ram_regions(
                    &mut self,
                ) -> Result<Vec<GuestRamRegion>, GuestRamRegionsError> {
                    self.0.set(self.0.get() + 1);
                    Ok(vec![GuestRamRegion {
                        gpa_base: 0,
                        host_va: 0x7f00_0000_0000,
                        len: 0x8000,
                    }])
                }
            }
            let mut host = Counting(std::cell::Cell::new(0));
            for _ in 0..5 {
                reference(&mut host, 0x1000, 8).expect("inside");
            }
            assert_eq!(host.0.get(), 1);
            assert_eq!(imports().len(), 1);
        });
    }

    /// A host with no import capability refuses every reference by name, and
    /// never asks the shim at all. This is the ordinary state on a host without
    /// the extension, so it must not read as a failure of the guest's work.
    #[test]
    fn without_a_published_granularity_nothing_is_asked_and_nothing_resolves() {
        with_granularity(None, || {
            let mut host = two_spans();
            assert_eq!(
                reference(&mut host, 0x2000, 8).err(),
                Some(MapRefusal::NoBackendImport)
            );
            assert!(imports().is_empty());
        });
    }

    /// The shim's own refusal is carried through with its name, rather than
    /// collapsing into "no imports". "This machine has no RAM span" and "this
    /// build's shim is too old" are different things to go fix.
    #[test]
    fn the_hosts_own_refusal_keeps_its_name() {
        with_granularity(Some(0x1000), || {
            assert_eq!(
                reference(&mut Refusing, 0x2000, 8).err(),
                Some(MapRefusal::HostRefused(GuestRamRegionsError::NoRam))
            );
        });
    }

    /// An address in the hole between two spans is refused by name. It is not a
    /// bound violation — no import claims it — and reporting it as one would
    /// send a reader looking for arithmetic that is not there.
    #[test]
    fn an_address_no_span_backs_is_refused_by_name() {
        with_granularity(Some(0x1000), || {
            let mut host = two_spans();
            let hole = 0x8000_0000;
            assert_eq!(
                reference(&mut host, hole, 8).err(),
                Some(MapRefusal::GpaNotInAnyImport { gpa: hole })
            );
        });
    }

    /// A length that leaves the span it started in is refused by the bound, with
    /// the bound's own reason. The next span's bytes are elsewhere in host
    /// memory, so running off the end of one import is exactly the stray this
    /// device is bounded against.
    #[test]
    fn a_length_that_runs_off_the_end_of_a_span_is_refused_by_the_bound() {
        with_granularity(Some(0x1000), || {
            let mut host = two_spans();
            let last_page = 0x8000_0000 - 0x1000;
            assert!(reference(&mut host, last_page, 0x1000).is_ok());
            assert!(matches!(
                reference(&mut host, last_page, 0x2000),
                Err(MapRefusal::OutsideImport(
                    GuestRamError::SliceEndPastImport { .. }
                ))
            ));
        });
    }

    /// A span nothing can bound is skipped, and the ones that can be are still
    /// imported. Failing the whole map on one odd sliver would put a machine
    /// with an ordinary RAMBlock beside it on the copying rails for no reason.
    #[test]
    fn one_unusable_span_does_not_cost_the_usable_ones() {
        with_granularity(Some(0x1000), || {
            let mut host = Spans(vec![
                // Shorter than one granule: nothing to import.
                GuestRamRegion {
                    gpa_base: 0,
                    host_va: 0x7f00_0000_0000,
                    len: 0x400,
                },
                GuestRamRegion {
                    gpa_base: 0x1_0000_0000,
                    host_va: 0x7f80_0000_0000,
                    len: 0x8000,
                },
            ]);
            assert!(reference(&mut host, 0x1_0000_0000, 8).is_ok());
            assert_eq!(imports().len(), 1);
        });
    }

    /// Every span unusable is its own refusal, distinct from the host refusing:
    /// the host answered fine and our own bound rejected all of it.
    #[test]
    fn every_span_unusable_is_a_refusal_of_its_own() {
        with_granularity(Some(0x1000), || {
            let mut host = Spans(vec![GuestRamRegion {
                gpa_base: 0,
                host_va: 0x7f00_0000_0000,
                len: 0x400,
            }]);
            assert_eq!(
                reference(&mut host, 0, 8).err(),
                Some(MapRefusal::NoUsableRegion { spans: 1 })
            );
        });
    }

    /// A device recreate drops the imports, and the rebuilt ones carry new
    /// identities. A reference taken before the teardown must not resolve
    /// against the replacement, because the backend handle it named is gone.
    #[test]
    fn a_reset_rebuilds_against_fresh_identities() {
        with_granularity(Some(0x1000), || {
            let mut host = two_spans();
            let before = reference(&mut host, 0x2000, 8).expect("inside");
            reset();
            let after = reference(&mut host, 0x2000, 8).expect("inside");
            assert_ne!(before.import().id(), after.import().id());
            assert!(matches!(
                after
                    .import()
                    .resolve(&before.import().slice(0, 8).unwrap()),
                Err(GuestRamError::SliceForeignImport { .. })
            ));
        });
    }

    /// The refusals reach the always-on log, and the expected one does not
    /// reach the fail channel.
    #[test]
    fn refusals_are_visible_and_the_expected_one_is_not_a_failure() {
        with_granularity(Some(0x1000), || {
            let capture = crate::observe::FailCapture::start();
            let mut host = two_spans();
            let _ = reference(&mut host, 0x8000_0000, 8);
            let line = capture.one(EVENT);
            assert!(
                line.contains("reason=guest_ram_map_gpa_not_in_any_import"),
                "{line}"
            );
            assert!(line.contains("gpa=0x80000000"), "{line}");
            assert!(!line.starts_with("OFF "), "a lost reference is a failure");
        });
        with_granularity(None, || {
            let capture = crate::observe::FailCapture::start();
            let mut host = two_spans();
            let _ = reference(&mut host, 0x2000, 8);
            // Not `capture.one(EVENT)`: an off-channel line's first token is
            // the literal `OFF`, which is the same thing the fail-log reading
            // notes in `AGENTS.md` warn about when ranking `reason=`.
            let lines = capture.lines();
            assert_eq!(
                lines,
                vec!["OFF guest_ram_map reason=guest_ram_map_no_backend_import"],
                "a host without the extension has not lost guest work"
            );
        });
    }

    /// The `runs` a `Scattered` refusal reports is the coalescer's answer, not
    /// a second count that agrees with it today.
    ///
    /// This number decides whether widening the bind to N ranges is worth
    /// building, and how expensive it would be — a window in two stretches and
    /// a window in five hundred both refuse identically without it. So it is
    /// asserted against `reims_vgpu_paging::runs::contig_run_count` on the same input rather
    /// than against a literal: a hand-written expectation here would be a
    /// second implementation of the coalescing rule, and the one that drifts is
    /// always the copy nothing else reads.
    #[test]
    fn a_scattered_refusal_reports_the_run_count_the_coalescer_finds() {
        const PAGE: u64 = 4096;
        // Nine pages in four stretches: 3 + 1 + 2 + 3.
        let gpas: Vec<u64> = vec![
            0x1000, 0x2000, 0x3000, // run 1
            0x9000, // run 2
            0x20000, 0x21000, // run 3
            0x50000, 0x51000, 0x52000, // run 4
        ];
        let expected = reims_vgpu_paging::runs::contig_run_count(&gpas, PAGE);
        assert_eq!(expected, 4, "fixture must actually be four stretches");

        with_granularity(Some(PAGE), || {
            let capture = crate::observe::FailCapture::start();
            let mut host = two_spans();
            let err = reference_for_pages(&mut host, &gpas, PAGE, 0, 8).unwrap_err();
            assert_eq!(
                err,
                MapRefusal::Scattered {
                    pages: gpas.len(),
                    runs: expected,
                    first: 0x1000,
                }
            );
            let line = capture.one(EVENT);
            assert!(line.contains("reason=guest_ram_map_scattered"), "{line}");
            assert!(
                line.contains("runs=4"),
                "the run count reaches the log: {line}"
            );
            assert!(line.contains("pages=9"), "{line}");
        });

        // One stretch is not scattered, so it must not refuse for this reason —
        // otherwise `runs` would be reporting on a population that includes the
        // case the widening is supposed to leave alone.
        let contiguous: Vec<u64> = (0..4).map(|i| 0x1000 + i * PAGE).collect();
        assert_eq!(
            reims_vgpu_paging::runs::contig_run_count(&contiguous, PAGE),
            1
        );
        with_granularity(Some(PAGE), || {
            let mut host = two_spans();
            let out = reference_for_pages(&mut host, &contiguous, PAGE, 0, 8);
            assert!(
                !matches!(out, Err(MapRefusal::Scattered { .. })),
                "one contiguous stretch must not refuse as scattered"
            );
        });
    }

    /// The runs tile the requested window exactly: ascending, no gap, no
    /// overlap, and summing to the length asked for.
    ///
    /// This is the property that keeps a scattered writeback from corrupting
    /// guest memory, and every part of it is a real failure mode rather than a
    /// tidiness rule. A gap leaves a band of the guest's surface holding the
    /// previous frame while the rest updates. An overlap writes one stretch of
    /// guest RAM twice from two different source offsets, so the winner is
    /// whichever copy the GPU retires last. A sum short of `len` is a torn
    /// frame; a sum past it is a write beyond the window the caller was given
    /// — which the import's own bound would catch, but only after the plan had
    /// already decided to make it.
    ///
    /// Asserted as a walk over the returned runs rather than as expected
    /// tuples, so the test states the invariant instead of restating one
    /// fixture's arithmetic.
    #[test]
    fn scattered_runs_tile_the_window_exactly() {
        const PAGE: u64 = 4096;
        // Four stretches, deliberately uneven: 3 + 1 + 2 + 3 pages.
        let gpas: Vec<u64> = vec![
            0x1000, 0x2000, 0x3000, // run 1
            0x9000, // run 2
            0x20000, 0x21000, // run 3
            0x50000, 0x51000, 0x52000, // run 4
        ];
        // Start part-way into the first page and end part-way into the last, so
        // the head and tail clips are both exercised. Neither end is page
        // aligned, which is the normal case for a sample window.
        let in_page = 100u64;
        let len = PAGE * 8 + 55;

        with_granularity(Some(PAGE), || {
            let mut host = two_spans();
            let runs = references_for_runs(&mut host, &gpas, PAGE, in_page, len)
                .expect("a scattered window still resolves");
            assert_eq!(runs.len(), 4, "one reference per contiguous stretch");

            let mut expected_offset = 0u64;
            for run in &runs {
                assert_eq!(
                    run.window_offset, expected_offset,
                    "runs must be ascending and leave no gap"
                );
                let bound = run.guest.bound().expect("each run resolves in its import");
                // `requested` is what the caller asked for; `bound_len` may be
                // larger because the import rounds to its granularity. The tiling
                // is a statement about the former.
                expected_offset += run.guest.requested();
                assert!(bound.len >= run.guest.requested());
            }
            assert_eq!(
                expected_offset, len,
                "the runs together must cover exactly the window asked for"
            );
        });
    }

    /// A window inside one stretch yields one run, so the widened path is not a
    /// different answer for the case that already worked.
    ///
    /// Worth pinning because the two entry points now compute the same thing by
    /// different routes: a divergence would mean a contiguous surface landed
    /// differently depending on which rail asked, which is exactly the "two arms
    /// consume one wire form" class.
    #[test]
    fn a_contiguous_window_is_one_run_and_agrees_with_the_single_reference() {
        const PAGE: u64 = 4096;
        let gpas: Vec<u64> = (0..4).map(|i| 0x1000 + i * PAGE).collect();
        with_granularity(Some(PAGE), || {
            let mut host = two_spans();
            let runs = references_for_runs(&mut host, &gpas, PAGE, 64, PAGE * 3)
                .expect("contiguous resolves");
            assert_eq!(runs.len(), 1);
            assert_eq!(runs[0].window_offset, 0);

            let single = reference_for_pages(&mut host, &gpas, PAGE, 64, PAGE * 3)
                .expect("and so does the single-range entry point");
            assert_eq!(
                runs[0].guest.bound().expect("bound"),
                single.bound().expect("bound"),
                "one stretch must bind identically whichever entry point asked"
            );
        });
    }

    /// A host with no import refuses the widened path by the same name, on the
    /// same channel, as the narrow one.
    ///
    /// The widened path calls `reference` once per run, so a naive
    /// implementation reports the host-wide refusal once per run — five hundred
    /// identical lines for one fact about the machine. `report_once` is what
    /// stops that, and this asserts it still does through the new caller.
    #[test]
    fn no_backend_import_stays_one_line_through_the_widened_path() {
        const PAGE: u64 = 4096;
        let gpas: Vec<u64> = vec![0x1000, 0x2000, 0x9000, 0x20000];
        with_granularity(None, || {
            let capture = crate::observe::FailCapture::start();
            let mut host = two_spans();
            let err = references_for_runs(&mut host, &gpas, PAGE, 0, PAGE).unwrap_err();
            assert_eq!(err, MapRefusal::NoBackendImport);
            let lines = capture.lines();
            assert_eq!(
                lines,
                vec!["OFF guest_ram_map reason=guest_ram_map_no_backend_import"],
                "one statement about the host, not one per run"
            );
        });
    }

    /// Both page-list entry points answer a host with no import the same way.
    ///
    /// They consume one wire form — a decoded page list — and until the
    /// `standing_refusal` gate went in they disagreed about a host, not about
    /// their input: `references_for_runs` reached `reference` and got
    /// `NoBackendImport` on the off channel, while `reference_for_pages`
    /// judged contiguity first and put `guest_ram_map_scattered` on the
    /// **fail** channel — a claim of lost guest work naming a cause that was
    /// not the cause. A driven `REIMS_VGPU_GUEST_IMPORT=off` boot logged
    /// exactly one of those lines, which is what this pins.
    ///
    /// The assertion is that the two arms agree, not that either says a
    /// particular thing, because a divergence is what the fail log cannot
    /// survive.
    #[test]
    fn both_page_list_arms_name_the_host_and_not_the_window() {
        const PAGE: u64 = 4096;
        // Scattered on purpose: this is the input that used to reach the
        // contiguity refusal before anything asked whether an import existed.
        let gpas: Vec<u64> = vec![0x1000, 0x2000, 0x9000, 0x20000];
        with_granularity(None, || {
            let capture = crate::observe::FailCapture::start();
            let mut host = two_spans();

            let narrow = reference_for_pages(&mut host, &gpas, PAGE, 0, PAGE).unwrap_err();
            let wide = references_for_runs(&mut host, &gpas, PAGE, 0, PAGE).unwrap_err();
            assert_eq!(
                narrow, wide,
                "one page list, one host, two entry points: the reason cannot depend on which asked"
            );
            assert_eq!(narrow, MapRefusal::NoBackendImport);

            assert_eq!(
                capture.lines(),
                vec!["OFF guest_ram_map reason=guest_ram_map_no_backend_import"],
                "a host that cannot import has not lost guest work to fragmentation"
            );
        });
    }
}
