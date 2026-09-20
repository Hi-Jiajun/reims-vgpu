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
use std::collections::BTreeSet;
use std::sync::{Mutex, OnceLock};

use metal_api_core::provider::{
    disposition_retires_resources, AllocationId, BufferLease, CompletionDisposition, ContractError,
    DeviceEpoch, GuestWindow, GuestWindows, HostRegion, LeaseId, LeaseImporter, LeaseLedger,
    LeaseObservation, LeaseReservation, NoCopyLeaseImporter, ProviderError, StagedLease,
};
use metal_api_vulkan::VulkanComputeProvider;

use crate::observe::Decline as ObserveDecline;

/// `size_of::<GuestRun>()` as the run-list meters' own unit: the seventh cut
/// prices a run list by its elements, so `_bytes / GUEST_RUN_BYTES` is the run
/// count the round read.
const GUEST_RUN_BYTES: u64 = std::mem::size_of::<metal_api_core::provider::GuestRun>() as u64;

/// What one device-loss teardown released, as the owner ledger saw it.
///
/// The contract's guarantee is that a device loss releases every lease
/// regardless of outstanding tokens (`ProviderHealth::DeviceLost`,
/// `research/docs/13` §3.2), so this census is the evidence that the teardown
/// ran against real state rather than an empty ledger: it is carried in the
/// refuse log line and asserted by the injection test.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceLossTeardown {
    /// Leases the ledger held when the teardown started.
    pub leases: usize,
    /// Windows the teardown retired and reclaimed.
    pub windows: usize,
}

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

/// The bytes one staged request states, in the two arms the plan can mint a
/// lease from.
///
/// # Why the arm is stated by the caller rather than decided by the plan
///
/// `StagedLease::new` wants the bytes **owned** — the provider holds them until
/// the completion that uploads them — so a submission has to end up with a
/// `Vec<u8>` of them either way. What this type records is *which* `Vec` that
/// is: the one this rail already made in the class gate ([`Self::Owned`]), or a
/// copy the plan still has to make out of a borrow ([`Self::Borrowed`]).
///
/// The owner of the bytes is the only code that can answer that, which is why
/// the arm travels in the request instead of being re-derived here: the gate
/// holds a window's copy in [`WindowCopies`], and a `Vec` it has handed over
/// cannot be read again.
///
/// The switch (`REIMS_VGPU_STAGED_BYTES_OWNED`, off by default) decides whether
/// a caller that *does* hold such a `Vec` hands it over; with the switch off
/// every request is [`Self::Borrowed`] and the plan's copy is the one it always
/// made.
#[derive(Clone, Debug)]
pub enum StagedBytes<'a> {
    /// The `Vec` this rail already holds, moved into the lease.
    ///
    /// The bytes are the ones the class gate copied for a window the device's
    /// granules turned away (`R18`) or read out of the guest's live pages for a
    /// run list (`G1-A`), so a lease minted from this arm is minted around the
    /// **same** `Vec` the submission would otherwise have copied a second time
    /// and then dropped.
    Owned(Vec<u8>),
    /// A borrow of bytes this rail does not own: the plan copies them, which is
    /// the arm every round before the seventh cut ran.
    Borrowed(&'a [u8]),
}

impl StagedBytes<'_> {
    /// Bytes this request states, whichever arm carries them.
    fn len(&self) -> u64 {
        match self {
            Self::Owned(bytes) => u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            Self::Borrowed(bytes) => u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        }
    }
}

/// One binding whose bytes the owner holds for the duration of this call: the
/// staged copy this rail already gathered, with no registered window behind it.
///
/// Not `Copy` since the seventh cut: the [`StagedBytes::Owned`] arm carries the
/// `Vec` itself, so a request that states it is the only one that can.
#[derive(Clone, Debug)]
pub struct Staged<'a> {
    pub binding: u32,
    pub bytes: StagedBytes<'a>,
}

/// One binding whose bytes are an ordered list of registered windows
/// (`research/docs/23` §74/§113, E-TX6).
///
/// The list arm exists because the caller's real source is not always one
/// window: a guest surface's bytes live in the pages the guest owns, and those
/// pages are described as runs — a head, then a stretch of pages, then a tail —
/// never as one pointer the owner may hand over. Each window here is one such
/// run, in window order, and their concatenation *is* the binding's byte range.
///
/// The plan answers this request with the per-run coordinates of the contract's
/// `BufferSource::GuestRuns` ([`Plan::guest_runs`]) over the same one-lease-per-
/// registration grouping the single-window arm uses. Every window must name the
/// same registration: the contract pairs each run's reservation with the
/// declaring view's own allocation, so a list split across two is a shape it
/// cannot state ([`Decline::RunsSpanRegistrations`]).
#[derive(Clone, Copy, Debug)]
pub struct Runs<'a> {
    /// Canonical binding index the bytes belong to.
    pub binding: u32,
    /// The binding's windows, in the order the bytes are read. Every window
    /// carries its own `head`/`bytes_len`: the window is the page-aligned range
    /// the registration derived, and the binding's bytes start `head` bytes into
    /// it.
    pub windows: &'a [Window],
}

/// What one narrow-class binding offers the owner rail.
#[derive(Clone, Debug)]
pub enum Request<'a> {
    /// The bytes came from one registered guest RAM window; they may be
    /// imported without copying when the device supports it.
    Window(Window),
    /// The bytes are a staged copy the owner holds across the submission.
    Staged(Staged<'a>),
    /// The bytes are an ordered list of registered windows (E-TX6): the plan
    /// imports their registrations like any other window and states the
    /// contract's guest-runs arm over them.
    Runs(Runs<'a>),
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

impl Channel {
    /// The word this channel is reported under.
    ///
    /// A method rather than a `Debug` formatting at each site: the two names
    /// are read off a log line beside a lease id, and `Borrowed` against
    /// `borrowed` is the kind of difference a grep for one of them misses.
    pub fn name(self) -> &'static str {
        match self {
            Self::Borrowed => "borrowed",
            Self::Staged => "staged",
        }
    }
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
    /// A window carried no bytes to copy (`R18`): the staged arm it would
    /// become has nothing to hold, so it is refused rather than imported empty.
    WindowEmpty { binding: u32 },
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
    /// A guest-runs request stated no windows at all: a list whose
    /// concatenation is a byte range is never empty.
    RunsEmpty { binding: u32 },
    /// A guest-runs request's windows do not all live in one registration. The
    /// contract keys a run list on the *declaring view's* own allocation, so a
    /// list split across two registrations is a shape it cannot state
    /// (`research/docs/23` §113).
    RunsSpanRegistrations {
        binding: u32,
        first: u64,
        second: u64,
    },
}

impl ObserveDecline for Decline {
    fn slug(&self) -> &'static str {
        match self {
            Self::RegionInvalid { .. } => "owner_region_invalid",
            Self::DuplicateRegion { .. } => "owner_duplicate_region",
            Self::UnregisteredRegion { .. } => "owner_unregistered_region",
            Self::RegionRetired { .. } => "owner_region_retired",
            Self::WindowOutsideRegion { .. } => "owner_window_outside_region",
            Self::WindowEmpty { .. } => "owner_window_empty",
            Self::HostImportUnavailable { .. } => "owner_host_import_unavailable",
            Self::LeaseImport { .. } => "owner_lease_import",
            Self::LeaseRelease { .. } => "owner_lease_release",
            Self::LeaseNotHeld { .. } => "owner_lease_not_held",
            Self::CompletionNotRetiring { .. } => "owner_completion_not_retiring",
            Self::CompletionWithoutToken { .. } => "owner_completion_without_token",
            Self::Ledger { .. } => "owner_ledger",
            Self::WindowRegistry { .. } => "owner_window_registry",
            Self::StagedEmpty { .. } => "owner_staged_empty",
            Self::RunsEmpty { .. } => "owner_runs_empty",
            Self::RunsSpanRegistrations { .. } => "owner_runs_span_registrations",
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
            Self::WindowEmpty { binding } => vec![("binding", binding.to_string())],
            Self::LeaseImport { binding, detail } => {
                vec![("binding", binding.to_string()), ("detail", detail.clone())]
            }
            Self::LeaseRelease { binding, detail } => {
                vec![("binding", binding.to_string()), ("detail", detail.clone())]
            }
            Self::LeaseNotHeld { binding }
            | Self::CompletionNotRetiring { binding }
            | Self::CompletionWithoutToken { binding }
            | Self::StagedEmpty { binding }
            | Self::RunsEmpty { binding } => vec![("binding", binding.to_string())],
            Self::RunsSpanRegistrations {
                binding,
                first,
                second,
            } => vec![
                ("binding", binding.to_string()),
                ("first_import", first.to_string()),
                ("second_import", second.to_string()),
            ],
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
    retired: BTreeSet<u64>,
    ledger: LeaseLedger,
    windows: GuestWindows,
    /// Lease ids whose window this rail registered, kept because the provider's
    /// own `GuestWindows` type has no iterator: a device-loss teardown must
    /// retire and reclaim every window explicitly (`research/docs/20` §3.4
    /// step 3) rather than silently dropping the registry that exists to keep
    /// an active window's backing alive.
    window_leases: BTreeSet<u64>,
    /// How many device-loss teardowns this rail has run, and the census of the
    /// most recent one. Both are read back by the fail log and by tests.
    teardowns: usize,
    last_teardown: DeviceLossTeardown,
    next_allocation: u64,
    next_lease: u64,
}

impl State {
    fn new() -> Self {
        Self {
            regions: BTreeMap::new(),
            retired: BTreeSet::new(),
            ledger: LeaseLedger::new(),
            windows: GuestWindows::new(),
            window_leases: BTreeSet::new(),
            teardowns: 0,
            last_teardown: DeviceLossTeardown::default(),
            next_allocation: 1,
            next_lease: 1,
        }
    }

    /// Register one window and remember its lease, so a later teardown can
    /// retire and reclaim it by name.
    fn register_window(&mut self, window: GuestWindow) -> Result<(), ContractError> {
        self.windows.register(window)?;
        self.window_leases.insert(window.lease.get());
        Ok(())
    }

    /// Retire and reclaim one window, in the contract's order
    /// (`research/docs/20` §3.4 step 3), and forget its lease.
    fn reclaim_window(&mut self, lease: LeaseId) -> Result<(), ContractError> {
        self.windows.retire(lease)?;
        self.windows.reclaim(lease)?;
        self.window_leases.remove(&lease.get());
        Ok(())
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

/// Every import this rail has retired, in identity order.
///
/// The reading half of [`State::retired`], which is otherwise only visible as
/// the `owner_region_retired` refusal on the next submission: a regression test
/// asserts that an address rule retiring a RAMBlock-shaped import never lands
/// here (`research/docs/20` §3.4 step 4), and a census can say how many
/// registrations have stopped deriving windows.
pub fn retired_regions() -> Vec<u64> {
    lock().retired.iter().copied().collect()
}

/// One window's own bytes, read out of the registration that names it (`R18`).
///
/// The owner rail's *copy* answer for a window its device cannot import: the
/// render rail states a window-backed bind as the borrowed no-copy arm when the
/// view's own host pointer is a whole number of the device's import granules,
/// and as an owner-issued staged lease over these bytes when it is not. The
/// bytes are the ones the borrowed arm would have bound — `host_va + head` for
/// `bytes_len` bytes — so the two arms differ in *how* the provider gets them
/// and not in which bytes the declaration reads.
///
/// Every check the read needs comes from this rail's own state, in one critical
/// section with the read itself: the import has to be registered here, its
/// address rail must not have retired it, the window has to cover exactly the
/// bind's bytes, and the window has to be a range inside the registration. The
/// registration is what bounds the read — it was built by [`register`] from
/// this process's own mapping of the import and validated by the provider's
/// `HostRegion::validate` — so no window a caller invents can point this at
/// bytes outside it.
///
/// `Err` is a typed refusal (`owner_unregistered_region`,
/// `owner_region_retired`, `owner_window_outside_region`, `owner_window_empty`)
/// rather than an empty answer, so the caller can name the fact that stopped
/// the copy instead of inventing one.
pub fn window_bytes(window: Window) -> Result<Vec<u8>, Decline> {
    let state = lock();
    let record = state
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
    if window.bytes_len == 0 {
        return Err(Decline::WindowEmpty {
            binding: window.binding,
        });
    }
    let region = record.region;
    let region_start = u64::try_from(region.host_pointer).unwrap_or(u64::MAX);
    let region_end = region_start.checked_add(region.length);
    // The window has to be inside the registration *and* its `head` has to leave
    // the bind's own bytes inside the window: the view is what the declaration
    // reads, so the two coordinates are one condition.
    let window_inside = region_end.is_some_and(|end| {
        window.host_va >= region_start
            && window
                .host_va
                .checked_add(window.length)
                .is_some_and(|window_end| window_end <= end)
    });
    let bind_inside = window
        .head
        .checked_add(window.bytes_len)
        .is_some_and(|touched| touched <= window.length);
    if !window_inside || !bind_inside {
        return Err(Decline::WindowOutsideRegion {
            import: window.import,
            binding: window.binding,
            offset: window.host_va.saturating_sub(region_start),
            length: window.length,
            region_length: region.length,
            detail: format!(
                "head={} bytes_len={} region_start={region_start}",
                window.head, window.bytes_len
            ),
        });
    }
    let start = window
        .host_va
        .checked_add(window.head)
        .expect("the window covers the bind's bytes, so the view pointer is addressable");
    let len = usize::try_from(window.bytes_len).expect("the window is inside a host mapping");
    // SAFETY: `register` was handed this process's own mapping of the import
    // (`runtime::guest_ram_map`'s `register_owner_regions` projects
    // `GuestRamImport::host_base`), the checks above put the whole read inside
    // that registration, and an import the address rail retired was answered
    // above rather than read — a retired mapping is the one way these bytes
    // stop being this process's to read (`retire_region`). The mapping is held
    // for the import's lifetime, which is longer than this call: the copy is
    // made before the submission that names it.
    let bytes = unsafe { std::slice::from_raw_parts(start as *const u8, len) };
    Ok(bytes.to_vec())
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
    let _ = teardown(&mut state);
    state.teardowns = 0;
    state.last_teardown = DeviceLossTeardown::default();
}

/// Apply the provider's device-loss teardown guarantee to this rail and report
/// what it released.
///
/// `ProviderHealth::DeviceLost` is the one event that releases every lease
/// regardless of outstanding tokens (`research/docs/13` §3.2), and
/// `research/docs/20` §3.4 step 3 is the order the owner owes the windows: the
/// ledger marks the loss first, then every window is retired and reclaimed, and
/// only then is the rail's state replaced. Registrations are dropped as well:
/// every one of them was imported under the dead incarnation's epoch, and
/// re-registration after a rebuild is the recovery path's job. Identity
/// counters are deliberately not reset (`reset`'s reason).
///
/// Idempotent: a second call on an already-torn-down rail releases nothing and
/// reports a zero census, so a caller may run it on every refusal that names a
/// device loss without checking first.
pub fn teardown_device_lost() -> DeviceLossTeardown {
    let mut state = lock();
    let census = teardown(&mut state);
    state.teardowns += 1;
    state.last_teardown = census;
    census
}

/// The teardown body both entry points share: what was held, then the
/// contract's release order.
fn teardown(state: &mut State) -> DeviceLossTeardown {
    let census = DeviceLossTeardown {
        leases: state.ledger.leased().len(),
        windows: state.window_leases.len(),
    };
    for lease in std::mem::take(&mut state.window_leases) {
        let lease = LeaseId::new(lease);
        let _ = state.windows.retire(lease);
        let _ = state.windows.reclaim(lease);
    }
    debug_assert!(
        state.windows.is_empty(),
        "every registered window is tracked in `window_leases`"
    );
    state.ledger.device_lost();
    state.ledger = LeaseLedger::new();
    state.windows = GuestWindows::new();
    state.regions.clear();
    state.retired.clear();
    census
}

/// How many device-loss teardowns this rail has run.
pub fn device_loss_teardowns() -> usize {
    lock().teardowns
}

/// The census of the most recent device-loss teardown, zero before the first.
pub fn last_device_loss_teardown() -> DeviceLossTeardown {
    lock().last_teardown
}

/// The address rail announced that the mapping behind `import` is gone.
///
/// Only an alias-shaped registration is announced here. The seam that
/// announces it (`crate::runtime::guest_ram_map::reclaim_alias`) decides on the
/// guest-RAM ledger's own `gpa_base` first, because an address rule never ends
/// a RAMBlock registration (`research/docs/20` §3.4 step 4): the boot pass owns
/// it for the VM's lifetime and every later narrow-class window is cut from it.
///
/// The registration stays — a late window or reclaim is then answered by name
/// — but it derives no further window. A window still active under it is left
/// for its own completion to retire and reclaim (`research/docs/20` §3.4: a
/// host range must not be handed back underneath work the GPU may still be
/// doing); one whose lease observation already retired it is reclaimed here, so
/// a settled window cannot outlive the registration that backed it.
pub fn retire_region(import: u64) {
    let mut state = lock();
    retire_region_in(&mut state, import);
}

/// [`retire_region`] against a state the caller already holds, so a unit test
/// can drive the window/lease half without the process-global rail.
fn retire_region_in(state: &mut State, import: u64) {
    // The same answer `reclaim_alias` decided on, asked once more against the
    // shape this rail holds. Taking an address retirement for a RAMBlock-shaped
    // registration would refuse every later window over it
    // (`owner_region_retired`) while the guest-RAM ledger still resolved — the
    // double-ledger disagreement a boot measured while the narrow class was the
    // first thing to cut a window from an imported block
    // (`evidence/gate3-zerocopy-narrow-2026-09-17`, import=9). A release build
    // must not carry it, so the skip is unconditional rather than an assertion.
    if state
        .regions
        .get(&import)
        .is_some_and(|record| record.region.gpa_base.is_some())
    {
        return;
    }
    state.retired.insert(import);
    let lease_id = state.regions.get(&import).map(|record| record.lease_id);
    if let Some(lease_id) = lease_id {
        let lease = LeaseId::new(lease_id);
        if state.windows.is_reclaimable(lease) {
            let _ = state.reclaim_window(lease);
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
    /// One entry per [`Request::Runs`] the plan answered, in request order.
    run_views: Vec<RunView>,
}

/// One binding's guest-runs list, with the lease identity the contract's
/// per-run checks need (`research/docs/23` §74/§113, E-TX6).
#[derive(Debug)]
struct RunView {
    binding: u32,
    /// The lease's allocation: the registration the runs were cut from. The
    /// contract pairs every run's reservation with the *declaring view's* own
    /// allocation (`LeaseMismatch` otherwise), so this is also the allocation
    /// the view that carries the list has to name.
    allocation: AllocationId,
    allocation_size: u64,
    reservation: LeaseReservation,
    runs: Vec<metal_api_core::provider::GuestRun>,
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

    /// One binding's guest-runs list, in the order the request stated it, with
    /// each run's coordinates inside the lease's own allocation
    /// (`research/docs/23` §74/§113, E-TX6).
    ///
    /// `None` is a binding whose bytes the plan did not state as a list. The
    /// runs are what a `BufferSource::GuestRuns` declaration carries; the
    /// allocation they are stated inside is [`Self::run_allocation`].
    pub fn guest_runs(&self, binding: u32) -> Option<&[metal_api_core::provider::GuestRun]> {
        self.run_views
            .iter()
            .find(|view| view.binding == binding)
            .map(|view| view.runs.as_slice())
    }

    /// The lease allocation one binding's guest-runs list lives in.
    ///
    /// The contract requires every run's reservation to name the declaring
    /// view's own allocation, so the caller states its view over this pair
    /// rather than over an identity it minted itself.
    pub fn run_allocation(&self, binding: u32) -> Option<(AllocationId, u64, u64)> {
        self.run_views
            .iter()
            .find(|view| view.binding == binding)
            .map(|view| {
                (
                    view.allocation,
                    view.allocation_size,
                    view.reservation.offset,
                )
            })
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
        // A list-only binding has no `View` of its own — its lease is the one
        // its windows' registration already group into — so a submission whose
        // only owner-backed input is a run list would otherwise hand the trace
        // no allocation record for the lease it names.
        for view in &self.run_views {
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
    ///
    /// The owner ledger decides each lease's fate first (`abandon_hold` in this
    /// module): a lease the ledger still holds — bound to a token that never
    /// retired — keeps its window and its provider import, so the two records
    /// cannot disagree about who owns the backing.
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
                state.window_leases.remove(&window.lease.get());
            }
            if state.ledger.release(hold.lease).is_none() {
                return Err(Decline::LeaseNotHeld {
                    binding: hold.binding,
                });
            }
            release_import(provider, hold)?;
            hold.released = true;
            // The retirement chain, in the order it ran, once per lease. This
            // is the `research/docs/20` §3.4 evidence a report about a
            // completed no-copy submission is written from, and it was the
            // half of the chain with no line at all.
            crate::observe::off(format!(
                "provider_owner_release lease={} channel={} reason=settled window={} import=released",
                hold.lease.get(),
                hold.channel.name(),
                if hold.window.is_some() {
                    "retired+reclaimed"
                } else {
                    "none"
                },
            ));
        }
        Ok(())
    }

    fn abort_inner(&mut self, provider: &VulkanComputeProvider) {
        let mut state = lock();
        for hold in &mut self.holds {
            if hold.released {
                continue;
            }
            abandon_hold(&mut state, provider, hold);
        }
    }
}

/// The owner ledger's half of [`abandon_hold`]: release the lease and answer
/// whether the provider import may follow.
///
/// `false` means the ledger still holds the lease — it is bound to a token
/// that never retired, and the contract names no release for it short of a
/// device-loss teardown (`research/docs/20` §3.4). The caller must then keep
/// the import held as well, so the ledger and the provider cannot end up
/// disagreeing about who owns the backing (`S1`,
/// `evidence/reviews/reims-gate2-increment23-review-2026-09-17.md` §3).
fn release_abandoned_lease(state: &mut State, hold: &Hold) -> bool {
    state.ledger.release(hold.lease).is_some()
}

/// Give up one abandoned hold as an explicit pair: the owner ledger releases
/// the lease first, the window it registered is retired and reclaimed second,
/// and only then is the provider import dropped. The ledger is the gate: a
/// lease it refuses to release keeps its window *and* its import, because the
/// import is the provider's record of the same lease.
///
/// Returns whether the hold is now given up.
fn abandon_hold(state: &mut State, provider: &VulkanComputeProvider, hold: &mut Hold) -> bool {
    if !release_abandoned_lease(state, hold) {
        // The kept half, named. A lease bound to a token that never retired has
        // no release short of a device-loss teardown, and the window and the
        // provider import stay with it; a line here is what says the rail chose
        // that on purpose rather than leaking it.
        crate::observe::off(format!(
            "provider_owner_lease_kept lease={} channel={} reason=ledger_still_holds \
             (the window and the provider import are kept with it)",
            hold.lease.get(),
            hold.channel.name(),
        ));
        return false;
    }
    if let Some(window) = hold.window {
        // A window that never reached a completion is retired by the owner's
        // own decision, which is the only authority left once the submission
        // is abandoned. It is retired only after the ledger gave the lease up,
        // in the same order `settle_inner` keeps.
        let _ = state.reclaim_window(window.lease);
    }
    if release_import(provider, hold).is_err() {
        // The import is still held by the provider; the line is emitted by the
        // caller's own decline, and a second failure here would only overwrite
        // it.
        return false;
    }
    hold.released = true;
    crate::observe::off(format!(
        "provider_owner_release lease={} channel={} reason=abandoned window={} import=released",
        hold.lease.get(),
        hold.channel.name(),
        if hold.window.is_some() {
            "retired+reclaimed"
        } else {
            "none"
        },
    ));
    true
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
    requests: &mut [Request<'a>],
) -> Result<Plan, Decline> {
    plan_with_epoch(provider, provider.device_epoch().get(), requests)
}

/// One window, checked against the registration it names: the byte range it
/// covers inside that registration, as `(offset, end)`.
///
/// The single-window arm and the run-list arm ask exactly this question, and
/// the answer is what both group by — so it is spelled once. A window that
/// leaves its registration, one whose registration this rail does not hold or
/// has retired, and one whose head plus bytes leave the window are all refused
/// by the same names they always were.
fn checked_window(state: &State, window: Window) -> Result<(u64, u64), Decline> {
    let record = state
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
    let end = offset
        .checked_add(window.length)
        .ok_or(Decline::WindowOutsideRegion {
            import: window.import,
            binding: window.binding,
            offset,
            length: window.length,
            region_length: region.length,
            detail: "window end overflows".into(),
        })?;
    let touched =
        window
            .head
            .checked_add(window.bytes_len)
            .ok_or(Decline::WindowOutsideRegion {
                import: window.import,
                binding: window.binding,
                offset,
                length: window.length,
                region_length: region.length,
                detail: "view end overflows".into(),
            })?;
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
    Ok((offset, end))
}

/// The owner's bytes for one staged binding, in the arm the request stated.
///
/// # Why the plan is where the copy had to be made, and why it need not be
///
/// [`StagedLease::new`] wants the bytes **owned** — the provider holds them
/// until the completion that uploads them, which is later than this call — so a
/// submission ends up holding a `Vec<u8>` either way. What the seventh cut
/// changed is where that `Vec` comes from:
///
/// - [`StagedBytes::Owned`]: the caller handed over the `Vec` it had already
///   built (the class gate's copy of a window the granules turned away, or the
///   bytes a gather read out of the guest's live pages). This arm **moves** it,
///   and the copy the plan used to make — on the calling thread, with the
///   registry guard of [`plan_with_epoch`] alive — is gone.
/// - [`StagedBytes::Borrowed`]: the caller only has a borrow, so the owner's
///   `Vec` is made here exactly as it was before the cut.
///
/// The two arms hand `StagedLease::new` the same bytes: the move is a move of
/// the buffer the copy would have been taken from.
///
/// Every reading is default off behind the frame profile, and both arms are
/// read under the same two meters — `LeaseVecMeter::StagedCopy` for the bytes
/// that had to be copied and `LeaseVecMeter::StagedMoved` for the bytes that
/// did not — so one round says which arm ran and how much it carried. With
/// `REIMS_VGPU_FRAME_PROFILE` unset this function is `bytes.to_vec()` for a
/// borrowed request and the identity for an owned one, and no clock is read.
fn staged_lease_bytes(bytes: StagedBytes<'_>) -> Vec<u8> {
    use crate::runtime::drain::{frame_span, note_lease_vec, FrameSpan, LeaseVecMeter};
    let _span = frame_span(FrameSpan::LeaseStagedCopy);
    match bytes {
        StagedBytes::Owned(bytes) => {
            note_lease_vec(
                LeaseVecMeter::StagedMoved,
                1,
                u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            );
            bytes
        }
        StagedBytes::Borrowed(bytes) => {
            let copy = bytes.to_vec();
            // Reported before the value moves into `StagedLease::new`, so the
            // counters and the span are of the same construction.
            note_lease_vec(
                LeaseVecMeter::StagedCopy,
                1,
                u64::try_from(copy.len()).unwrap_or(u64::MAX),
            );
            copy
        }
    }
}

/// Whether `REIMS_VGPU_STAGED_BYTES_OWNED` asked a caller that holds a staged
/// binding's `Vec` to hand it over instead of lending a slice of it.
///
/// **Off by default, and off is the arm every round before the seventh cut
/// ran**: with the switch off the requests are all [`StagedBytes::Borrowed`]
/// and [`staged_lease_bytes`] makes the copy it always made, byte for byte.
/// Set `REIMS_VGPU_STAGED_BYTES_OWNED=1` (also `on`, `true`, `yes`) for the arm
/// that moves the caller's own buffer into the lease.
///
/// Read once, at the first call, like the rail's other mechanism switches: the
/// answer cannot change inside a process, and the gate sits on a path that runs
/// once per staged binding.
pub(crate) fn staged_bytes_owned_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_staged_bytes_owned(
            std::env::var("REIMS_VGPU_STAGED_BYTES_OWNED")
                .ok()
                .as_deref(),
        )
    })
}

/// The switch's own parser, apart from the process-global it caches into, so a
/// unit test can read every spelling without a switch it cannot put back.
fn parse_staged_bytes_owned(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "on" | "true" | "yes")
    )
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
    requests: &mut [Request<'a>],
) -> Result<Plan, Decline> {
    let alignment = provider.no_copy_alignment();
    let mut state = lock();
    let mut holds: Vec<Hold> = Vec::new();
    let mut views: Vec<View> = Vec::new();
    let mut run_views: Vec<RunView> = Vec::new();

    // Window requests are grouped by registration first: one lease per
    // registration, covering the union of the windows this submission touches.
    let mut groups: BTreeMap<u64, (u64, u64, u32)> = BTreeMap::new();
    let mut staged: Vec<(u32, u64)> = Vec::new();
    // The run-list requests, in request order, beside the windows they group
    // into. A list is not a view of its own: its runs are stated inside the
    // lease its registration already mints, which is why the grouping above and
    // this list are one answer rather than two.
    let mut run_requests: Vec<Runs<'_>> = Vec::new();
    for request in requests.iter() {
        match request {
            Request::Window(window) => {
                channel(window.binding, true, alignment)?;
                let (offset, end) = checked_window(&state, *window)?;
                let entry = groups
                    .entry(window.import)
                    .or_insert((offset, end, window.binding));
                entry.0 = entry.0.min(offset);
                entry.1 = entry.1.max(end);
            }
            Request::Staged(staged_request) => {
                channel(staged_request.binding, false, alignment)?;
                let length = staged_request.bytes.len();
                if length == 0 {
                    return Err(Decline::StagedEmpty {
                        binding: staged_request.binding,
                    });
                }
                staged.push((staged_request.binding, length));
            }
            // The list arm (`research/docs/23` §113, E-TX6). Every window is
            // checked exactly as a single-window request's is, and the grouping
            // is the same one: the runs of one registration share the one lease
            // this request's registrations mint. What the arm adds is that the
            // windows are also kept, in the order the request stated them, so
            // the coordinates can be stated per run below.
            //
            // A list whose windows span two registrations is refused here: the
            // contract pairs every run's reservation with the *declaring
            // view's* own allocation, so no single declaration can state a list
            // that lives in two. (The engine's own run walk does split at an
            // import seam, so this is a real shape and not a hypothetical one —
            // it keeps the engine by name rather than being stated as a list
            // the provider would refuse.)
            Request::Runs(run_request) => {
                let Some((first, rest)) = run_request.windows.split_first() else {
                    return Err(Decline::RunsEmpty {
                        binding: run_request.binding,
                    });
                };
                channel(run_request.binding, true, alignment)?;
                let (offset, end) = checked_window(&state, *first)?;
                let mut union = (offset, end);
                for window in rest {
                    if window.import != first.import {
                        return Err(Decline::RunsSpanRegistrations {
                            binding: run_request.binding,
                            first: first.import,
                            second: window.import,
                        });
                    }
                    channel(window.binding, true, alignment)?;
                    let (offset, end) = checked_window(&state, *window)?;
                    union.0 = union.0.min(offset);
                    union.1 = union.1.max(end);
                }
                let entry =
                    groups
                        .entry(first.import)
                        .or_insert((union.0, union.1, run_request.binding));
                entry.0 = entry.0.min(union.0);
                entry.1 = entry.1.max(union.1);
                run_requests.push(*run_request);
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
        if let Err(error) = state.register_window(window) {
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
            let _ = state.reclaim_window(lease);
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
        // The success path, named. A refusal has always been emitted — by the
        // caller of `plan` for the `Decline` this returns — and the import that
        // *worked* was silent, which is the half a report about the no-copy
        // rail actually needs: "which window, how big, and did it go in
        // without a copy" cannot be answered from a log that only has the
        // failures. Emitted once per lease, so a narrow-class dispatch adds a
        // handful of lines and the copying path adds the same handful with
        // `no_copy=0`.
        crate::observe::off(format!(
            "provider_owner_lease channel=borrowed no_copy=1 lease={} import={} binding={} \
             window_offset={} window_length={} pages={} region_bytes={}",
            lease.get(),
            import,
            binding,
            offset,
            end - offset,
            (end - offset) / alignment.max(1),
            region.length,
        ));
        holds.push(Hold {
            channel: Channel::Borrowed,
            binding: *binding,
            lease,
            window: Some(window),
            released: false,
        });
        for request in requests.iter() {
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
        // The run lists that live in this registration, in the order the
        // requests stated them. The coordinates are the contract's own
        // (`research/docs/23` §113, E-TX6): each run names the lease that
        // registration minted and its own `(offset, length)` inside the
        // *allocation* — the same namespace the single-window views above state
        // — so the provider's per-run bound check reads the reservation this
        // loop just registered.
        let region_pointer = u64::try_from(region.host_pointer).unwrap_or(u64::MAX);
        for run_request in &run_requests {
            if run_request.windows.first().map(|w| w.import) != Some(*import) {
                continue;
            }
            // The seventh cut's own meter: one fresh `Vec<GuestRun>` per
            // list-shaped request, built here under the registry guard. The
            // bar and the two counters are default off behind the frame
            // profile; with it off this is the same `Vec::with_capacity` and
            // the same pushes.
            let _run_span =
                crate::runtime::drain::frame_span(crate::runtime::drain::FrameSpan::LeaseRunGather);
            let mut runs = Vec::with_capacity(run_request.windows.len());
            for window in run_request.windows {
                let Some(offset) = window
                    .host_va
                    .checked_sub(region_pointer)
                    .and_then(|offset| offset.checked_add(window.head))
                else {
                    // Unreachable behind `checked_window`, and unwound anyway:
                    // a plan that cannot build every run must not leave the
                    // imports it already made behind.
                    return Err(abort_with(
                        provider,
                        &mut state,
                        holds,
                        Decline::WindowOutsideRegion {
                            import: *import,
                            binding: run_request.binding,
                            offset: 0,
                            length: window.length,
                            region_length: region.length,
                            detail: "run offset overflows".into(),
                        },
                    ));
                };
                runs.push(metal_api_core::provider::GuestRun {
                    lease_id: lease,
                    offset,
                    length: window.bytes_len,
                });
            }
            crate::runtime::drain::note_lease_vec(
                crate::runtime::drain::LeaseVecMeter::RunGather,
                1,
                u64::try_from(runs.len())
                    .unwrap_or(u64::MAX)
                    .saturating_mul(GUEST_RUN_BYTES),
            );
            run_views.push(RunView {
                binding: run_request.binding,
                allocation,
                allocation_size: region.length,
                reservation,
                runs,
            });
        }
    }

    // The staged leases, one per binding that has no registered window.
    for (binding, length) in staged {
        let lease = LeaseId::new(state.lease());
        let allocation = AllocationId::new(state.allocate());
        // The request is asked *mutably* for its bytes because the arm that
        // owns them hands the `Vec` over rather than lending it: a lease minted
        // around the caller's own buffer is a lease the submission does not pay
        // to copy. The borrow arm is left in place by the same swap, and the
        // empty slice it is replaced with states nothing — the loop below is the
        // only reader of either arm.
        let bytes = requests
            .iter_mut()
            .find_map(|request| match request {
                Request::Staged(staged) if staged.binding == binding => Some(std::mem::replace(
                    &mut staged.bytes,
                    StagedBytes::Borrowed(&[]),
                )),
                _ => None,
            })
            .expect("a staged binding was collected above");
        let bytes = staged_lease_bytes(bytes);
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
            bytes,
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
        crate::observe::off(format!(
            "provider_owner_lease channel=staged no_copy=0 lease={} bytes={} binding={}",
            lease.get(),
            length,
            binding,
        ));
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
    // One line per plan, so a boot can be read for "how much of this
    // submission went in without copying" without walking the lease lines
    // above: a plan whose windows are all borrowed is the narrow class on a
    // registered guest-RAM block, and one whose bindings are all staged is the
    // same class with no window behind its bytes.
    let borrowed = holds
        .iter()
        .filter(|hold| hold.channel == Channel::Borrowed)
        .count();
    crate::observe::off(format!(
        "provider_owner_plan leases={} borrowed={} staged={}",
        holds.len(),
        borrowed,
        holds.len() - borrowed,
    ));
    Ok(Plan {
        holds,
        views,
        run_views,
    })
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
        run_views: Vec::new(),
    };
    for hold in &mut plan.holds {
        abandon_hold(state, provider, hold);
    }
    decline
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal_api_core::provider::{CompletionToken, SubmissionId};

    /// A plan hold with no window: the shape a staged binding produces, and
    /// everything [`release_abandoned_lease`] needs.
    fn hold(lease: LeaseId) -> Hold {
        Hold {
            channel: Channel::Staged,
            binding: 0,
            lease,
            window: None,
            released: false,
        }
    }

    fn reservation(lease: LeaseId, allocation: u64) -> LeaseReservation {
        LeaseReservation {
            lease: BufferLease {
                lease_id: lease,
                allocation_id: AllocationId::new(allocation),
                owner_epoch: DeviceEpoch::new(1),
            },
            offset: 0,
            length: 64,
        }
    }

    /// One registration projected into a local state with a window over it, so
    /// a test can drive the retirement chain without the process-global rail.
    /// Returns the lease the window carries.
    fn registered_window(state: &mut State, import: u64, gpa_base: Option<u64>) -> LeaseId {
        const PAGE: usize = 4096;
        let lease_id = state.lease();
        let region = Region {
            import,
            epoch: 1,
            host_pointer: PAGE,
            length: 2 * PAGE as u64,
            page_size: PAGE as u64,
            gpa_base,
        };
        let host = HostRegion {
            lease_id: LeaseId::new(lease_id),
            owner_epoch: DeviceEpoch::new(1),
            host_pointer: region.host_pointer,
            length: region.length,
            page_size: region.page_size,
        };
        state.regions.insert(
            import,
            Record {
                region,
                lease_id,
                host,
            },
        );
        let allocation = state.allocate();
        state
            .register_window(GuestWindow {
                lease: LeaseId::new(lease_id),
                allocation_id: AllocationId::new(allocation),
                offset: 0,
                length: PAGE as u64,
            })
            .expect("a window inside its registration");
        LeaseId::new(lease_id)
    }

    /// The window/lease half of `research/docs/20` §3.4 step 4, with no device:
    /// an address retirement never ends a RAMBlock-shaped registration, and an
    /// alias retirement leaves a still-active window to its own completion.
    ///
    /// An active window under a retired alias is deliberately not reclaimed
    /// here: only the completion that retires its lease may hand the host range
    /// back (the GPU may still be reading it). A window whose lease observation
    /// already retired it has no such claim left, so the retirement takes it
    /// rather than leaving it behind.
    #[test]
    fn an_address_retirement_skips_a_ramblock_shaped_registration() {
        let mut state = State::new();
        let _ramblock = registered_window(&mut state, 12, Some(0x1_0000_0000));
        let _alias = registered_window(&mut state, 11, None);

        retire_region_in(&mut state, 12);
        assert!(
            !state.retired.contains(&12),
            "a RAMBlock-shaped registration keeps deriving windows"
        );
        assert!(
            state.regions.contains_key(&12),
            "and keeps its registration"
        );
        assert_eq!(state.windows.len(), 2, "and keeps the window it had");

        retire_region_in(&mut state, 11);
        assert!(
            state.retired.contains(&11),
            "the alias stops deriving windows"
        );
        assert!(
            state.regions.contains_key(&11),
            "its registration stays: a late window or reclaim is answered by name"
        );
        assert_eq!(
            state.windows.len(),
            2,
            "an active window is retired by its own completion, not by a mapping change"
        );

        // The settled half: once the completion retired the window and no
        // reclaim has taken it yet, the retirement reclaims it rather than
        // leaving a host range the lease observation has already given back.
        let mut state = State::new();
        let alias = registered_window(&mut state, 11, None);
        state
            .windows
            .retire(alias)
            .expect("the window is registered");
        retire_region_in(&mut state, 11);
        assert!(state.retired.contains(&11));
        assert!(
            state.windows.is_empty(),
            "the retired window is reclaimed with its registration"
        );
        assert!(
            state.window_leases.is_empty(),
            "and its lease is forgotten with it"
        );
    }

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

    /// The list arm's own two shape refusals, by name and with the fields a
    /// census line needs (`research/docs/23` §113, E-TX6): a request with no
    /// windows at all, and one whose windows name two registrations — which no
    /// single declaration can state, because the contract pairs every run's
    /// reservation with the declaring view's own allocation.
    #[test]
    fn a_run_list_the_contract_cannot_state_is_refused_by_its_own_name() {
        let empty = Decline::RunsEmpty { binding: 0x50000 };
        assert_eq!(empty.slug(), "owner_runs_empty");
        assert_eq!(empty.fields(), vec![("binding", "327680".to_string())]);

        let split = Decline::RunsSpanRegistrations {
            binding: 0x50000,
            first: 9,
            second: 11,
        };
        assert_eq!(split.slug(), "owner_runs_span_registrations");
        assert_eq!(
            split.fields(),
            vec![
                ("binding", "327680".to_string()),
                ("first_import", "9".to_string()),
                ("second_import", "11".to_string()),
            ],
            "the refusal names both registrations, so the log says which pair"
        );
    }

    /// The window check the single-window and run-list arms share: the byte
    /// range a window covers inside its registration, and the three refusals
    /// that keep a window from being grouped at all.
    #[test]
    fn a_window_is_checked_against_the_registration_it_names() {
        const PAGE: usize = 4096;
        let mut state = State::new();
        registered_window(&mut state, 21, Some(0x2_0000_0000));
        let window = |host_va: u64, length: u64, head: u64, bytes_len: u64| Window {
            binding: 0x50000,
            import: 21,
            host_va,
            length,
            head,
            bytes_len,
        };
        // The registration's own first granule, read from 4 bytes in: the pair
        // the run list states for its first run.
        assert_eq!(
            checked_window(&state, window(PAGE as u64, PAGE as u64, 4, 64)),
            Ok((0, PAGE as u64)),
            "an in-registration window answers its own range"
        );
        let past = checked_window(&state, window(PAGE as u64, 4 * PAGE as u64, 0, 64))
            .expect_err("a window past the registration");
        assert_eq!(past.slug(), "owner_window_outside_region");
        let before = checked_window(&state, window(0, PAGE as u64, 0, 64))
            .expect_err("a window before the registration");
        assert_eq!(before.slug(), "owner_window_outside_region");
        let unregistered = checked_window(
            &state,
            Window {
                import: 22,
                ..window(PAGE as u64, PAGE as u64, 0, 64)
            },
        )
        .expect_err("a window naming a registration this rail does not hold");
        assert_eq!(unregistered.slug(), "owner_unregistered_region");
        // A view that reaches past its own window: the head plus the bytes the
        // binding reads is the range the provider would bind.
        let reaching = checked_window(&state, window(PAGE as u64, PAGE as u64, PAGE as u64, 64))
            .expect_err("a view past its window");
        assert_eq!(reaching.slug(), "owner_window_outside_region");
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

    /// S1: an abort releases a provider import only as the explicit pair of an
    /// owner-ledger release. The unreachable arm — a lease bound to a token
    /// that never retired — is locked here structurally: the ledger keeps the
    /// lease, so its import stays held with it instead of the ledger claiming a
    /// lease the provider has already released.
    #[test]
    fn an_abandoned_release_is_paired_with_the_owner_ledger() {
        let mut state = State::new();
        let lease = LeaseId::new(91);
        state
            .ledger
            .register(reservation(lease, 9_001))
            .expect("a legal reservation");

        // Never bound: the ledger gives the lease up, so its provider import
        // may be released with it.
        assert!(
            release_abandoned_lease(&mut state, &hold(lease)),
            "the ledger releases an unbound lease"
        );
        assert!(
            !state.ledger.contains(lease),
            "and the lease leaves the ledger in the same step"
        );

        // Bound to a token that never retired: the ledger refuses, and the
        // import must stay held — the contract names no release for a bound
        // lease short of a device-loss teardown.
        let bound = LeaseId::new(92);
        state
            .ledger
            .register(reservation(bound, 9_002))
            .expect("a legal reservation");
        state
            .ledger
            .bind(
                bound,
                CompletionToken {
                    submission_id: SubmissionId::new(4),
                    device_epoch: DeviceEpoch::new(1),
                },
            )
            .expect("a valid token binds the lease");
        assert!(
            !release_abandoned_lease(&mut state, &hold(bound)),
            "a still-bound lease is not the abort's to release"
        );
        assert!(
            state.ledger.contains(bound),
            "the ledger still holds the lease, so the caller keeps its import"
        );
        assert!(
            !state.ledger.release_ready(bound),
            "and the lease is not release-ready until its token retires"
        );
    }

    /// The seventh cut's two arms hand the lease the same bytes.
    ///
    /// The cut is a move of the caller's own buffer instead of a copy of a
    /// borrow, so the reading it has to survive is byte-level: whatever
    /// `StagedLease::new` is given must be the bytes the request stated. The
    /// copy arm is the arm every round before the cut ran; the move arm is the
    /// one the switch selects. Neither is allowed to change a byte, and the
    /// moved arm is not allowed to leave a copy behind (`to_vec` would be a
    /// second allocation the round would not see).
    #[test]
    fn the_staged_arms_hand_the_lease_the_same_bytes() {
        let bytes: Vec<u8> = (0..=255u8).cycle().take(4_096).collect();
        let borrowed = staged_lease_bytes(StagedBytes::Borrowed(&bytes));
        assert_eq!(borrowed, bytes, "the borrow arm copies the request's bytes");
        assert_ne!(
            borrowed.as_ptr(),
            bytes.as_ptr(),
            "the borrow arm's brace is the copy: it is not the caller's buffer"
        );

        // The move arm's own identity: the `Vec` the lease is minted around is
        // the one the caller handed over, pointer and all — which is what makes
        // it a move rather than a copy the round would not see.
        let handed = bytes.clone();
        let handed_ptr = handed.as_ptr();
        let moved = staged_lease_bytes(StagedBytes::Owned(handed));
        assert_eq!(
            moved.as_ptr(),
            handed_ptr,
            "the move arm hands the caller's own buffer to the lease"
        );
        assert_eq!(moved, bytes, "the move arm hands over the same bytes");
    }

    /// The switch is off for every spelling but the four that mean "on".
    ///
    /// Read apart from the process-global it caches into, so a test can read
    /// every spelling without a switch it cannot put back — the same shape the
    /// rail's other mechanism switches are read by.
    #[test]
    fn the_staged_bytes_switch_is_off_unless_it_is_asked_for() {
        for value in [
            None,
            Some(""),
            Some("off"),
            Some("0"),
            Some("no"),
            Some("ON!"),
            Some("tru"),
        ] {
            assert!(!parse_staged_bytes_owned(value), "{value:?} is not an arm");
        }
        for value in ["1", "on", "true", "yes", " ON ", "Yes"] {
            assert!(
                parse_staged_bytes_owned(Some(value)),
                "{value:?} asks for the handover arm"
            );
        }
    }
}
