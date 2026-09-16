//! Owner-side guest-RAM rail for the canonical provider.
//!
//! This module is the fork's half of `research/docs/20` step 5, the step the
//! guest-memory design calls "reims becomes the owner": the process that holds
//! the guest RAM mapping registers it in the provider's own `HostRegion` shape,
//! cuts the page-aligned window one submission actually touches out of that
//! registration, imports it through the provider's real lease API, and releases
//! it only after the completion that retired the GPU work. Nothing here is a
//! bookkeeping mirror the provider never sees.
//!
//! # What lives where
//!
//! - [`Region`] is one registration: a guest RAM import
//!   ([`crate::runtime::guest_ram_map`]) projected into the provider's
//!   `HostRegion` shape. [`register`] builds that type and runs its own
//!   `validate()`, so a page size, pointer alignment or length the provider
//!   would refuse is refused at the handshake rather than one dispatch later.
//! - [`Plan`] is one submission's lease set: one lease per registration a
//!   window was cut from, plus one staged lease per binding that has no
//!   registered window. [`plan`] imports every lease through the provider's
//!   `LeaseImporter`/`NoCopyLeaseImporter` calls and registers the window in the
//!   owner's `GuestWindows` and `LeaseLedger`.
//! - [`Plan::settle`] is the retirement chain of `research/docs/20` §3.4: bind
//!   the provider's completion token, observe that the completion retires the
//!   lease, retire and reclaim the window, release the provider import.
//!   [`Plan::abort`] is the same chain for a submission that never got a
//!   completion, so a declined dispatch cannot leave an import behind.
//!
//! # The device gate
//!
//! A binding whose bytes came from a registered guest RAM window needs the
//! device's host-pointer import (`VK_EXT_external_memory_host`, i.e. a nonzero
//! `no_copy_alignment()`). [`channel`] is that decision and it is deliberately
//! not a preference: a window-backed binding on a device without the extension
//! is refused ([`Decline::HostImportUnavailable`]) instead of being quietly
//! turned into a copy, because a silent copy is the one answer that would make
//! the owner rail look wired while its registrations were never consumed.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use metal_api_core::provider::{
    disposition_retires_resources, AllocationId, BufferLease, CompletionDisposition, DeviceEpoch,
    GuestWindow, GuestWindows, HostRegion, LeaseId, LeaseImporter, LeaseLedger, LeaseObservation,
    LeaseReservation, NoCopyLeaseImporter, ProviderError, StagedLease,
};
use metal_api_vulkan::VulkanComputeProvider;

use crate::observe::Decline as ObserveDecline;

/// One guest RAM region registered with the provider contract.
///
/// Every field is taken from the import this process holds — never from the
/// shim's raw answer — so a registration cannot name bytes the import does not
/// cover. `page_size` is the backend's measured import granularity, which is
/// also the only legal window alignment over the region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    /// Identity of the import this region was projected from
    /// (`ImportId::get`). The one thing a lease identity is derived from.
    pub import: u64,
    /// The device epoch the registration was made under.
    pub epoch: u64,
    /// First host address of the registration, already aligned to `page_size`.
    pub host_pointer: usize,
    /// Registration length, a whole number of `page_size` granules.
    pub length: u64,
    /// Import granularity the registration declares as its page size.
    pub page_size: u64,
    /// First guest physical address, or `None` for the packed-alias shape
    /// ([`crate::runtime::guest_ram_map`]'s `gpa_base`).
    pub gpa_base: Option<u64>,
}

/// One binding's window: the page-aligned host range its staged bytes came
/// from, inside the registration named by `import`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    /// Canonical binding index the bytes belong to.
    pub binding: u32,
    /// Registration (import identity) the window was cut from.
    pub import: u64,
    /// First host byte of the window.
    pub host_va: u64,
    /// Window length in bytes, a whole number of the registration's granules.
    pub length: u64,
    /// Bytes from the window's first byte to the first byte of the staged
    /// range the kernel actually reads or writes.
    pub head: u64,
    /// Bytes the kernel touches, from `head` onwards.
    pub bytes_len: u64,
}

/// One binding whose bytes the owner holds for the duration of this call: the
/// staged copy this rail already gathered, with no registered window behind it.
#[derive(Clone, Copy, Debug)]
pub struct Staged<'a> {
    pub binding: u32,
    pub bytes: &'a [u8],
}

/// What one narrow-class binding offers the owner rail.
#[derive(Clone, Copy, Debug)]
pub enum Request<'a> {
    /// The bytes came from one registered guest RAM window; they may be
    /// imported without copying when the device supports it.
    Window(Window),
    /// The bytes are a staged copy the owner holds across the submission.
    Staged(Staged<'a>),
}

/// Which owner channel one binding takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    /// The provider imports the owner mapping without copying
    /// (`BufferSource::BorrowedNoCopy`).
    Borrowed,
    /// The provider copies owner-issued bytes at import
    /// (`BufferSource::StagedLease`).
    Staged,
}

/// The owner channel one binding takes, or the device-gated refusal.
///
/// This is the whole capability gate, and the submission path calls it for
/// every binding: `has_window` is whether the binding's bytes came from a
/// registered guest RAM region, `host_import_alignment` is the device's
/// measured `no_copy_alignment()`. A window-backed binding on a device that
/// reports `0` is refused here — the owner rail must not turn "this device
/// cannot import host pointers" into a silent copy.
pub fn channel(
    binding: u32,
    has_window: bool,
    host_import_alignment: u64,
) -> Result<Channel, Decline> {
    match (has_window, host_import_alignment) {
        (true, 0) => Err(Decline::HostImportUnavailable { binding }),
        (true, _) => Ok(Channel::Borrowed),
        (false, _) => Ok(Channel::Staged),
    }
}

/// Why the owner rail refused a registration, an import or a retirement.
///
/// One variant per distinct check. The import/retirement arms carry the
/// provider's own slug in `detail` (e.g. `lease_epoch_mismatch`), so a log line
/// names both the owner step and the provider check inside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decline {
    /// The provider's `HostRegion::validate()` refused a registration built
    /// from this import: the projection is not a legal region for the contract.
    RegionInvalid { import: u64, detail: String },
    /// A second registration for one import identity.
    DuplicateRegion { import: u64 },
    /// A window named a registration this rail does not hold.
    UnregisteredRegion { import: u64, binding: u32 },
    /// The address rail announced the mapping moved; no window may be cut.
    RegionRetired { import: u64, binding: u32 },
    /// A window is not a page-aligned range inside the registration it names.
    WindowOutsideRegion {
        import: u64,
        binding: u32,
        offset: u64,
        length: u64,
        region_length: u64,
        detail: String,
    },
    /// The device reports no host-pointer import, so a window-backed binding
    /// cannot leave this rail: device-gated, fail-closed, no staged fallback.
    HostImportUnavailable { binding: u32 },
    /// The provider refused a lease import (identity, alignment or epoch).
    LeaseImport { binding: u32, detail: String },
    /// The provider refused a lease release.
    LeaseRelease { binding: u32, detail: String },
    /// A lease the owner expected to hold carried no outstanding binding.
    LeaseNotHeld { binding: u32 },
    /// The completion does not retire resources, so the lease must stay held.
    CompletionNotRetiring { binding: u32 },
    /// The completion carried no token to bind the lease to.
    CompletionWithoutToken { binding: u32 },
    /// The owner `LeaseLedger` refused a register/bind/observe/release step.
    Ledger { binding: u32, detail: String },
    /// The owner `GuestWindows` refused a register/retire/reclaim step.
    WindowRegistry { binding: u32, detail: String },
    /// A staged binding carried no bytes to import.
    StagedEmpty { binding: u32 },
}

impl ObserveDecline for Decline {
    fn slug(&self) -> &'static str {
        match self {
            Self::RegionInvalid { .. } => "owner_region_invalid",
            Self::DuplicateRegion { .. } => "owner_duplicate_region",
            Self::UnregisteredRegion { .. } => "owner_unregistered_region",
            Self::RegionRetired { .. } => "owner_region_retired",
            Self::WindowOutsideRegion { .. } => "owner_window_outside_region",
            Self::HostImportUnavailable { .. } => "owner_host_import_unavailable",
            Self::LeaseImport { .. } => "owner_lease_import",
            Self::LeaseRelease { .. } => "owner_lease_release",
            Self::LeaseNotHeld { .. } => "owner_lease_not_held",
            Self::CompletionNotRetiring { .. } => "owner_completion_not_retiring",
            Self::CompletionWithoutToken { .. } => "owner_completion_without_token",
            Self::Ledger { .. } => "owner_ledger",
            Self::WindowRegistry { .. } => "owner_window_registry",
            Self::StagedEmpty { .. } => "owner_staged_empty",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::RegionInvalid { import, detail } => {
                vec![("import", import.to_string()), ("detail", detail.clone())]
            }
            Self::DuplicateRegion { import } => vec![("import", import.to_string())],
            Self::UnregisteredRegion { import, binding } => vec![
                ("import", import.to_string()),
                ("binding", binding.to_string()),
            ],
            Self::RegionRetired { import, binding } => vec![
                ("import", import.to_string()),
                ("binding", binding.to_string()),
            ],
            Self::WindowOutsideRegion {
                import,
                binding,
                offset,
                length,
                region_length,
                detail,
            } => vec![
                ("import", import.to_string()),
                ("binding", binding.to_string()),
                ("offset", offset.to_string()),
                ("length", length.to_string()),
                ("region_length", region_length.to_string()),
                ("detail", detail.clone()),
            ],
            Self::HostImportUnavailable { binding } => {
                vec![("binding", binding.to_string())]
            }
            Self::LeaseImport { binding, detail } => {
                vec![("binding", binding.to_string()), ("detail", detail.clone())]
            }
            Self::LeaseRelease { binding, detail } => {
                vec![("binding", binding.to_string()), ("detail", detail.clone())]
            }
            Self::LeaseNotHeld { binding }
            | Self::CompletionNotRetiring { binding }
            | Self::CompletionWithoutToken { binding }
            | Self::StagedEmpty { binding } => vec![("binding", binding.to_string())],
            Self::Ledger { binding, detail } | Self::WindowRegistry { binding, detail } => {
                vec![("binding", binding.to_string()), ("detail", detail.clone())]
            }
        }
    }
}

crate::observe::decline_display!(Decline);

/// Render a provider refusal without a `Display` impl: its slug plus the
/// optional detail, which is where the provider names the refusing check.
fn provider_detail(error: &ProviderError) -> String {
    match &error.detail {
        Some(detail) => format!("{}: {detail}", error.slug),
        None => error.slug.clone(),
    }
}

/// One registration this rail holds, plus the lease identity a window over it
/// carries and the provider type the shape was validated as.
#[derive(Clone, Copy, Debug)]
struct Record {
    region: Region,
    /// The registration's lease identity. Windows over this region carry it, so
    /// the provider's own registry sees one import per registration at a time;
    /// a second concurrent window on the same region is refused by the
    /// provider (`lease_already_imported`) rather than silently shared.
    lease_id: u64,
    /// The registration as the provider contract sees it, kept so a test or a
    /// census can compare it field by field with the projection it came from.
    host: HostRegion,
}

#[derive(Default)]
struct State {
    regions: BTreeMap<u64, Record>,
    /// Imports the address rail retired: the registration stays (so a late
    /// question is answered by name) but derives no further window.
    retired: std::collections::BTreeSet<u64>,
    ledger: LeaseLedger,
    windows: GuestWindows,
    next_allocation: u64,
    next_lease: u64,
}

impl State {
    fn new() -> Self {
        Self {
            regions: BTreeMap::new(),
            retired: std::collections::BTreeSet::new(),
            ledger: LeaseLedger::new(),
            windows: GuestWindows::new(),
            next_allocation: 1,
            next_lease: 1,
        }
    }

    fn allocate(&mut self) -> u64 {
        let id = self.next_allocation;
        self.next_allocation += 1;
        id
    }

    fn lease(&mut self) -> u64 {
        let id = self.next_lease;
        self.next_lease += 1;
        id
    }
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(State::new()))
}

fn lock() -> std::sync::MutexGuard<'static, State> {
    state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Register one guest RAM region with the provider contract.
///
/// Builds the provider's own `HostRegion` from the projection and runs its
/// `validate()`, so the checks the provider would apply at an import (page
/// size, pointer alignment, length alignment, identities) run here instead, at
/// the handshake, against the same type. A refusal names the provider's own
/// error text; nothing is remembered on refusal.
pub fn register(region: Region) -> Result<(), Decline> {
    let mut state = lock();
    if state.regions.contains_key(&region.import) {
        return Err(Decline::DuplicateRegion {
            import: region.import,
        });
    }
    let lease_id = state.lease();
    let host = HostRegion {
        lease_id: LeaseId::new(lease_id),
        owner_epoch: DeviceEpoch::new(region.epoch),
        host_pointer: region.host_pointer,
        length: region.length,
        page_size: region.page_size,
    };
    host.validate().map_err(|error| Decline::RegionInvalid {
        import: region.import,
        detail: error.to_string(),
    })?;
    state.regions.insert(
        region.import,
        Record {
            region,
            lease_id,
            host,
        },
    );
    Ok(())
}

/// The registration this rail holds for `import`, as the provider contract
/// sees it, or `None` when no region was registered under that identity.
pub fn registered(import: u64) -> Option<HostRegion> {
    lock().regions.get(&import).map(|record| record.host)
}

/// How many registrations this rail holds. The census half of [`register`].
pub fn registered_regions() -> usize {
    lock().regions.len()
}

/// Forget every registration and drop every lease: the owner half of a device
/// recreate, called from the same place that drops the imports.
///
/// The provider's own teardown guarantee (`LeaseLedger::device_lost`) is
/// recorded before the ledger is replaced, so a lease that was still held when
/// the device died is released rather than looked up against the next
/// incarnation. Identity counters are deliberately **not** reset: a lease id
/// from the previous incarnation must never name a lease of the next one.
pub fn reset() {
    let mut state = lock();
    state.ledger.device_lost();
    state.ledger = LeaseLedger::new();
    state.windows = GuestWindows::new();
    state.regions.clear();
    state.retired.clear();
}

/// The address rail announced that the mapping behind `import` is gone.
///
/// The registration stays — a late window or reclaim is then answered by name
/// — but it derives no further window, and any window still registered under it
/// is retired so a later reclaim can take it.
pub fn retire_region(import: u64) {
    let mut state = lock();
    state.retired.insert(import);
    let lease_id = state.regions.get(&import).map(|record| record.lease_id);
    if let Some(lease_id) = lease_id {
        let lease = LeaseId::new(lease_id);
        if state.windows.is_reclaimable(lease) {
            let _ = state.windows.reclaim(lease);
        }
    }
}

/// One view of one binding inside the lease that backs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct View {
    pub binding: u32,
    /// Which owner channel imported the backing.
    pub channel: Channel,
    pub lease: LeaseId,
    pub allocation: AllocationId,
    /// Extent the resource table describes for the allocation: the whole
    /// registration for a borrowed window (`research/docs/20` §3.2's coordinate
    /// split), the staged bytes for a staged lease.
    pub allocation_size: u64,
    /// The reservation the lease registered, for the resource table snapshot.
    pub reservation: LeaseReservation,
    /// View offset inside that allocation.
    pub view_offset: u64,
    /// View length in bytes.
    pub view_length: u64,
}

/// One lease the rail imported, with everything the retirement chain needs.
#[derive(Clone, Copy, Debug)]
struct Hold {
    channel: Channel,
    /// The binding that names this lease in a log line.
    binding: u32,
    lease: LeaseId,
    window: Option<GuestWindow>,
    released: bool,
}

/// The leases one narrow submission imported, and the views that reference
/// them. Built by [`plan`]; consumed by [`Plan::settle`] or [`Plan::abort`].
#[derive(Debug)]
pub struct Plan {
    holds: Vec<Hold>,
    views: Vec<View>,
}

impl Plan {
    /// Every view, in binding order: what the trace builder needs.
    pub fn views(&self) -> &[View] {
        &self.views
    }

    /// The view for one canonical binding, if the plan imported a lease for it.
    pub fn view(&self, binding: u32) -> Option<&View> {
        self.views.iter().find(|view| view.binding == binding)
    }

    /// How many leases the plan holds. One per registration touched, plus one
    /// per staged binding.
    pub fn lease_count(&self) -> usize {
        self.holds.len()
    }

    /// Every lease the plan registered, deduplicated by lease: the allocation,
    /// the extent the resource table describes for it, and the reservation
    /// itself. A submission that touches two windows of one registration has
    /// two views over the entry returned once here.
    pub fn leases(&self) -> Vec<(AllocationId, u64, LeaseReservation)> {
        let mut out: Vec<(AllocationId, u64, LeaseReservation)> = Vec::new();
        for view in &self.views {
            if out.iter().any(|(_, _, reservation)| {
                reservation.lease.lease_id == view.reservation.lease.lease_id
            }) {
                continue;
            }
            out.push((view.allocation, view.allocation_size, view.reservation));
        }
        out
    }

    /// Retire and release every lease the plan holds, after the submission's
    /// completion. `completion` is what the provider returned from `submit`.
    ///
    /// The chain is `research/docs/20` §3.4, in order: bind the completion
    /// token to every lease, refuse a lease that reports no outstanding
    /// binding, refuse a completion that is not retirement evidence, observe
    /// the completion (which must retire), retire and reclaim each window, then
    /// release the provider's import. A refusal anywhere unwinds the rest
    /// through [`Plan::abort`], so a declined dispatch cannot leak an import.
    pub fn settle(
        mut self,
        provider: &VulkanComputeProvider,
        completion: CompletionDisposition,
    ) -> Result<(), Decline> {
        match self.settle_inner(provider, completion) {
            Ok(()) => Ok(()),
            Err(decline) => {
                self.abort_inner(provider);
                Err(decline)
            }
        }
    }

    /// Give up every lease the plan still holds. Called when the submission
    /// never reached a completion (a provider refusal, a writeback mismatch).
    pub fn abort(mut self, provider: &VulkanComputeProvider) {
        self.abort_inner(provider);
    }

    fn settle_inner(
        &mut self,
        provider: &VulkanComputeProvider,
        completion: CompletionDisposition,
    ) -> Result<(), Decline> {
        let first = self.holds.first().map(|hold| hold.binding).unwrap_or(0);
        let Some(token) = completion.token() else {
            return Err(Decline::CompletionWithoutToken { binding: first });
        };
        let mut state = lock();
        for hold in &self.holds {
            state
                .ledger
                .bind(hold.lease, token)
                .map_err(|error| Decline::Ledger {
                    binding: hold.binding,
                    detail: error.to_string(),
                })?;
        }
        for hold in &self.holds {
            // A lease with no outstanding binding after the bind above means the
            // ledger did not record what this plan just did; proceeding would
            // let the release through underneath work in flight.
            if state.ledger.release_ready(hold.lease) {
                return Err(Decline::LeaseNotHeld {
                    binding: hold.binding,
                });
            }
        }
        if !disposition_retires_resources(completion) {
            return Err(Decline::CompletionNotRetiring { binding: first });
        }
        for hold in &self.holds {
            let observation =
                state
                    .ledger
                    .observe(token, completion)
                    .map_err(|error| Decline::Ledger {
                        binding: hold.binding,
                        detail: error.to_string(),
                    })?;
            if observation != LeaseObservation::Retired {
                return Err(Decline::CompletionNotRetiring {
                    binding: hold.binding,
                });
            }
        }
        for hold in &self.holds {
            if let Some(window) = hold.window {
                state
                    .windows
                    .retire(window.lease)
                    .map_err(|error| Decline::WindowRegistry {
                        binding: hold.binding,
                        detail: error.to_string(),
                    })?;
            }
        }
        for hold in &mut self.holds {
            if let Some(window) = hold.window {
                state
                    .windows
                    .reclaim(window.lease)
                    .map_err(|error| Decline::WindowRegistry {
                        binding: hold.binding,
                        detail: error.to_string(),
                    })?;
            }
            if state.ledger.release(hold.lease).is_none() {
                return Err(Decline::LeaseNotHeld {
                    binding: hold.binding,
                });
            }
            release_import(provider, hold)?;
            hold.released = true;
        }
        Ok(())
    }

    fn abort_inner(&mut self, provider: &VulkanComputeProvider) {
        let mut state = lock();
        for hold in &mut self.holds {
            if hold.released {
                continue;
            }
            if let Some(window) = hold.window {
                // A window that never reached a completion is retired by the
                // owner's own decision, which is the only authority left once
                // the submission is abandoned.
                let _ = state.windows.retire(window.lease);
                let _ = state.windows.reclaim(window.lease);
            }
            let _ = state.ledger.release(hold.lease);
            if release_import(provider, hold).is_err() {
                // The import is still held by the provider; the line is emitted
                // by the caller's own decline, and a second failure here would
                // only overwrite it.
                continue;
            }
            hold.released = true;
        }
    }
}

/// Drop one lease from the provider that imported it.
fn release_import(provider: &VulkanComputeProvider, hold: &Hold) -> Result<(), Decline> {
    let answer = match hold.channel {
        Channel::Borrowed => provider.release_borrowed_lease(hold.lease),
        Channel::Staged => provider.release_staged_lease(hold.lease),
    };
    answer.map_err(|error| Decline::LeaseRelease {
        binding: hold.binding,
        detail: provider_detail(&error),
    })
}

/// Import every lease one narrow submission needs, through the provider's own
/// lease API.
///
/// Window-backed bindings of one registration share one lease: the provider
/// keys its import by lease id and one lease carries one reservation, so a
/// submission that touches two windows of one region imports their page-aligned
/// union and binds each view inside it (`research/docs/14`'s "one allocation,
/// many views"). Bindings with no registered window get their own staged lease.
///
/// A refusal anywhere unwinds every lease imported so far, so the caller sees
/// either a complete plan or no import at all.
pub fn plan<'a>(
    provider: &VulkanComputeProvider,
    requests: &[Request<'a>],
) -> Result<Plan, Decline> {
    plan_with_epoch(provider, provider.device_epoch().get(), requests)
}

/// [`plan`] with the lease epoch stated by the caller.
///
/// Production passes the provider's own `device_epoch()` — that is what
/// [`plan`] does — so the parameter exists for the one caller that must state a
/// *different* epoch on purpose: the stale-lease proof. A lease registered
/// under a previous device incarnation must be refused by the provider
/// (`lease_epoch_mismatch`) rather than resolving against this device, and that
/// refusal is only drivable with a real provider when the epoch can be named.
pub fn plan_with_epoch<'a>(
    provider: &VulkanComputeProvider,
    epoch: u64,
    requests: &[Request<'a>],
) -> Result<Plan, Decline> {
    let alignment = provider.no_copy_alignment();
    let mut state = lock();
    let mut holds: Vec<Hold> = Vec::new();
    let mut views: Vec<View> = Vec::new();

    // Window requests are grouped by registration first: one lease per
    // registration, covering the union of the windows this submission touches.
    let mut groups: BTreeMap<u64, (u64, u64, u32)> = BTreeMap::new();
    let mut staged: Vec<(u32, u64)> = Vec::new();
    for request in requests {
        match request {
            Request::Window(window) => {
                channel(window.binding, true, alignment)?;
                let record =
                    state
                        .regions
                        .get(&window.import)
                        .ok_or(Decline::UnregisteredRegion {
                            import: window.import,
                            binding: window.binding,
                        })?;
                if state.retired.contains(&window.import) {
                    return Err(Decline::RegionRetired {
                        import: window.import,
                        binding: window.binding,
                    });
                }
                let region = record.region;
                let offset = window
                    .host_va
                    .checked_sub(u64::try_from(region.host_pointer).unwrap_or(u64::MAX))
                    .ok_or(Decline::WindowOutsideRegion {
                        import: window.import,
                        binding: window.binding,
                        offset: 0,
                        length: window.length,
                        region_length: region.length,
                        detail: "window starts before the registration".into(),
                    })?;
                let end =
                    offset
                        .checked_add(window.length)
                        .ok_or(Decline::WindowOutsideRegion {
                            import: window.import,
                            binding: window.binding,
                            offset,
                            length: window.length,
                            region_length: region.length,
                            detail: "window end overflows".into(),
                        })?;
                let touched = window.head.checked_add(window.bytes_len).ok_or(
                    Decline::WindowOutsideRegion {
                        import: window.import,
                        binding: window.binding,
                        offset,
                        length: window.length,
                        region_length: region.length,
                        detail: "view end overflows".into(),
                    },
                )?;
                if end > region.length || touched > window.length {
                    return Err(Decline::WindowOutsideRegion {
                        import: window.import,
                        binding: window.binding,
                        offset,
                        length: window.length,
                        region_length: region.length,
                        detail: format!("end={end} touched={touched}"),
                    });
                }
                let entry = groups
                    .entry(window.import)
                    .or_insert((offset, end, window.binding));
                entry.0 = entry.0.min(offset);
                entry.1 = entry.1.max(end);
            }
            Request::Staged(staged_request) => {
                channel(staged_request.binding, false, alignment)?;
                if staged_request.bytes.is_empty() {
                    return Err(Decline::StagedEmpty {
                        binding: staged_request.binding,
                    });
                }
                staged.push((staged_request.binding, staged_request.bytes.len() as u64));
            }
        }
    }

    // The window leases. One registration, one lease, one provider import.
    for (import, (offset, end, binding)) in &groups {
        let record = state
            .regions
            .get(import)
            .copied()
            .expect("a grouped window was checked against its registration");
        let region = record.region;
        let lease = LeaseId::new(record.lease_id);
        let allocation = AllocationId::new(state.allocate());
        // The registration is re-stamped with this device's epoch: the epoch is
        // a property of the device incarnation, and the provider checks it on
        // every import (`lease_epoch_mismatch` otherwise).
        let host = HostRegion {
            owner_epoch: DeviceEpoch::new(epoch),
            ..record.host
        };
        let borrowed = host
            .borrowed_window(allocation, *offset, end - offset)
            .map_err(|error| Decline::WindowOutsideRegion {
                import: *import,
                binding: *binding,
                offset: *offset,
                length: end - offset,
                region_length: region.length,
                detail: error.to_string(),
            })?;
        let reservation = borrowed.reservation;
        if let Err(error) = state.ledger.register(reservation) {
            return Err(abort_with(
                provider,
                &mut state,
                holds,
                Decline::Ledger {
                    binding: *binding,
                    detail: error.to_string(),
                },
            ));
        }
        let window = GuestWindow {
            lease,
            allocation_id: allocation,
            offset: *offset,
            length: end - offset,
        };
        if let Err(error) = state.windows.register(window) {
            return Err(abort_with(
                provider,
                &mut state,
                holds,
                Decline::WindowRegistry {
                    binding: *binding,
                    detail: error.to_string(),
                },
            ));
        }
        // SAFETY: the registration proves the range is inside a host mapping
        // this process holds for the import's lifetime, the window was cut from
        // that registration through the provider's own checked derivation, and
        // the lease is released in `settle`/`abort` before this call returns.
        if let Err(error) = unsafe { provider.import_borrowed_lease(borrowed) } {
            let _ = state.ledger.release(lease);
            let _ = state.windows.retire(lease);
            let _ = state.windows.reclaim(lease);
            return Err(abort_with(
                provider,
                &mut state,
                holds,
                Decline::LeaseImport {
                    binding: *binding,
                    detail: provider_detail(&error),
                },
            ));
        }
        holds.push(Hold {
            channel: Channel::Borrowed,
            binding: *binding,
            lease,
            window: Some(window),
            released: false,
        });
        for request in requests {
            let Request::Window(request) = request else {
                continue;
            };
            if request.import != *import {
                continue;
            }
            let view_offset = match request
                .host_va
                .checked_sub(u64::try_from(region.host_pointer).unwrap_or(u64::MAX))
                .and_then(|offset| offset.checked_add(request.head))
            {
                Some(view_offset) => view_offset,
                None => {
                    // Unreachable behind the checks above, and unwound anyway:
                    // a plan that cannot build every view must not leave the
                    // imports it already made behind.
                    return Err(abort_with(
                        provider,
                        &mut state,
                        holds,
                        Decline::WindowOutsideRegion {
                            import: *import,
                            binding: request.binding,
                            offset: 0,
                            length: request.length,
                            region_length: region.length,
                            detail: "view offset overflows".into(),
                        },
                    ));
                }
            };
            views.push(View {
                binding: request.binding,
                channel: Channel::Borrowed,
                lease,
                allocation,
                allocation_size: region.length,
                reservation,
                view_offset,
                view_length: request.bytes_len,
            });
        }
    }

    // The staged leases, one per binding that has no registered window.
    for (binding, length) in staged {
        let lease = LeaseId::new(state.lease());
        let allocation = AllocationId::new(state.allocate());
        let bytes = requests
            .iter()
            .find_map(|request| match request {
                Request::Staged(staged) if staged.binding == binding => Some(staged.bytes),
                _ => None,
            })
            .expect("a staged binding was collected above");
        let staged_lease = match StagedLease::new(
            LeaseReservation {
                lease: BufferLease {
                    lease_id: lease,
                    allocation_id: allocation,
                    owner_epoch: DeviceEpoch::new(epoch),
                },
                offset: 0,
                length,
            },
            bytes.to_vec(),
        ) {
            Ok(staged_lease) => staged_lease,
            Err(error) => {
                return Err(abort_with(
                    provider,
                    &mut state,
                    holds,
                    Decline::LeaseImport {
                        binding,
                        detail: error.to_string(),
                    },
                ))
            }
        };
        let reservation = staged_lease.reservation;
        if let Err(error) = state.ledger.register(reservation) {
            return Err(abort_with(
                provider,
                &mut state,
                holds,
                Decline::Ledger {
                    binding,
                    detail: error.to_string(),
                },
            ));
        }
        if let Err(error) = provider.import_staged_lease(staged_lease) {
            let _ = state.ledger.release(lease);
            return Err(abort_with(
                provider,
                &mut state,
                holds,
                Decline::LeaseImport {
                    binding,
                    detail: provider_detail(&error),
                },
            ));
        }
        holds.push(Hold {
            channel: Channel::Staged,
            binding,
            lease,
            window: None,
            released: false,
        });
        views.push(View {
            binding,
            channel: Channel::Staged,
            lease,
            allocation,
            allocation_size: length,
            reservation,
            view_offset: 0,
            view_length: length,
        });
    }

    views.sort_by_key(|view| view.binding);
    Ok(Plan { holds, views })
}

/// Unwind every lease imported so far and hand back the refusal that stopped
/// the plan, so the caller sees one answer and no half-imported set.
fn abort_with(
    provider: &VulkanComputeProvider,
    state: &mut State,
    holds: Vec<Hold>,
    decline: Decline,
) -> Decline {
    let mut plan = Plan {
        holds,
        views: Vec::new(),
    };
    for hold in &mut plan.holds {
        if let Some(window) = hold.window {
            let _ = state.windows.retire(window.lease);
            let _ = state.windows.reclaim(window.lease);
        }
        let _ = state.ledger.release(hold.lease);
        if release_import(provider, hold).is_ok() {
            hold.released = true;
        }
    }
    decline
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_on_a_device_without_host_import_is_refused_by_name() {
        // The device-gated branch: the capability gate is the whole answer for a
        // window-backed binding, and it must be a named refusal rather than a
        // silent staged copy.
        let refusal = channel(7, true, 0).expect_err("no host-pointer import");
        assert_eq!(refusal.slug(), "owner_host_import_unavailable");
        assert_eq!(
            refusal.fields(),
            vec![("binding", "7".to_string())],
            "the refusal names the binding it refused"
        );
    }

    #[test]
    fn a_window_on_a_capable_device_and_a_staged_binding_take_their_channels() {
        assert_eq!(channel(1, true, 4096), Ok(Channel::Borrowed));
        assert_eq!(channel(1, false, 0), Ok(Channel::Staged));
        assert_eq!(channel(1, false, 4096), Ok(Channel::Staged));
    }

    #[test]
    fn a_registration_with_an_illegal_page_size_is_refused_by_the_provider_type() {
        reset();
        let page = 4096usize;
        let refusal = register(Region {
            import: 1_000_001,
            epoch: 1,
            host_pointer: page,
            length: 2 * page as u64,
            // Zero is not a power of two: the provider's own `validate()`
            // refuses it, and the refusal names that check.
            page_size: 0,
            gpa_base: Some(page as u64),
        })
        .expect_err("zero page size");
        assert_eq!(refusal.slug(), "owner_region_invalid");
        assert!(
            refusal.fields().iter().any(|(key, value)| *key == "detail"
                && value.contains("page size 0 must be a nonzero power of two")),
            "the provider's own error must be visible: {refusal:?}"
        );
        reset();
    }
}
