//! The sampled-texture statement payload census (task E-SW3, the statement
//! economy's W4).
//!
//! # What it answers, and why it is one instrument and not two
//!
//! W1's section account (`e_statement_accounting-report.md`) measured one
//! statement at 1.86 MB, of which **39.4% is texture payload** — the sampled
//! textures' own tightly packed extents, real bytes the provider reads. This
//! module answers the two questions a cut against that payload has to be
//! chosen on, in one place so the two readings cannot drift apart:
//!
//! 1. **What the payload is made of.** One counter pair per arm a sampled
//!    texture can be stated through ([`PayloadArm`]): the request's own copy,
//!    R24's registry frame, G1-B's windowless gather, R36's repacked rows,
//!    R41's folded plan, R28's window arms, the trace's own production, the
//!    pass-entry snapshot, the production restated by a later trace, and the
//!    compute rail's staged texture.
//! 2. **How much of it is a repeat.** Every byte-carrying declaration is
//!    classified as *first*, an **in-statement repeat** (bytes this same
//!    statement already carried, which a statement-local reference arm could
//!    serve with no cross-submission state at all) or an **earlier-statement
//!    repeat** (bytes an earlier statement carried, which is the class that
//!    owes a key and a generation a provider can check).
//!
//! The three classes are what say which cut the workload can pay for, and the
//! arms are what say which slice of the 39.4% each cut is against. A reading
//! that only had the arms could not tell a first copy from a repeat; one that
//! only had the classes could not tell which arm to cut.
//!
//! # Default off, and a reading rather than a shape
//!
//! `REIMS_VGPU_TEXTURE_PAYLOAD_CENSUS` (`1`/`on`/`true`/`yes`) arms it. Off,
//! every entry point is one relaxed load and no counter moves. Nothing here
//! changes a byte of any frame in either state: the walk reads the trace the
//! rail was about to state and the counters land in the census's own
//! `store_routes` window beside the routes the arms already charge.

use metal_api_core::provider::{ComputeTrace, TextureSource, TextureView, TracePass};

/// The arm one sampled texture's payload was stated by, as the declaring walk
/// knows it.
///
/// The names are the rail's own (`sampled_bind_arm`'s vocabulary) plus the two
/// sites outside that walk, so a reading here and a route in the same window
/// name one arm rather than two.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(usize)]
pub(crate) enum PayloadArm {
    /// [`crate::backend::provider_render`]'s `NarrowTextureSource::Bytes`: the
    /// request's own tightly packed copy.
    Trace = 0,
    /// R24's frame read out of the engine's registry.
    Frame = 1,
    /// G1-B's copy out of the bind's own live runs (no registration window).
    Gathered = 2,
    /// R36's repacked rows: the guest's padded window cutting one row at a
    /// time.
    Depadded = 3,
    /// R41's folded channel plan over the class gate's own texels.
    Folded = 4,
    /// R28's registered window, stated as a lease (the wire carries the lease,
    /// not the bytes).
    Window = 5,
    /// The trace's own production (`TextureSource::TraceView`: no payload).
    Produced = 6,
    /// The pass's own attachment (`TextureSource::PassEntrySnapshot`: no
    /// payload).
    Snapshot = 7,
    /// A production restated by a later trace: the window arm's bytes read out
    /// of the registration and stated as the trace's own copy.
    Restate = 8,
    /// The compute rail's staged texture (`provider_compute`'s own
    /// `TextureSource::OwnedBytes`).
    Compute = 9,
}

/// Number of [`PayloadArm`] slots, derived from the enum the same way the
/// other meter enums in this crate do.
pub(crate) const PAYLOAD_ARMS: usize = PayloadArm::Compute as usize + 1;

/// The `(count, bytes)` counter stems of each arm, in slot order.
const ARM_COUNTERS: [(&str, &str); PAYLOAD_ARMS] = [
    ("tex_payload_trace_n", "tex_payload_trace_bytes"),
    ("tex_payload_frame_n", "tex_payload_frame_bytes"),
    ("tex_payload_gathered_n", "tex_payload_gathered_bytes"),
    ("tex_payload_depadded_n", "tex_payload_depadded_bytes"),
    ("tex_payload_folded_n", "tex_payload_folded_bytes"),
    ("tex_payload_window_n", "tex_payload_window_bytes"),
    ("tex_payload_produced_n", "tex_payload_produced_bytes"),
    ("tex_payload_snapshot_n", "tex_payload_snapshot_bytes"),
    ("tex_payload_restate_n", "tex_payload_restate_bytes"),
    ("tex_payload_compute_n", "tex_payload_compute_bytes"),
];

/// The most payload identities one boot's statements may be counted under.
///
/// The same shape `COPY_IDENTITIES_MAX` has, and for the same reason: at the
/// cap the table stops admitting new identities instead of evicting one, so a
/// returning identity still reads as an earlier-statement repeat and the
/// identities that could not be filed are counted by name
/// (`tex_payload_boot_ids_overflow_n`) rather than silently under-reporting the
/// repeat rate.
const BOOT_IDS_MAX: usize = 65_536;

/// Whether this process is counting texture payloads, read once.
///
/// Read once per process, like every other switch in this crate: the census is
/// armed by a launcher before the first statement and never changes under a
/// running round.
pub(crate) fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        parse(
            std::env::var("REIMS_VGPU_TEXTURE_PAYLOAD_CENSUS")
                .ok()
                .as_deref(),
        )
    })
}

/// The arm override, the way `zero_fill_decl_arm` has one: a process that
/// already read the environment cannot be re-armed otherwise, and a rail test
/// that links this crate as a library has to be able to move the counters this
/// module charges.
///
/// `0` is "no override" and restores the environment's own answer; a round
/// never writes it.
static OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Arm or disarm the census in a test (`Some(true)` on), leaving the
/// environment's own answer alone.
///
/// Tests only; the runtime never calls it.
#[doc(hidden)]
pub fn set_texture_payload_census_for_test(arm: Option<bool>) {
    use std::sync::atomic::Ordering::Relaxed;
    OVERRIDE.store(
        match arm {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        Relaxed,
    );
}

/// [`enabled`] with a test's own override folded in. The runtime's entry
/// points read this one, so a test can move the counters without an
/// environment the rest of the process would keep.
#[inline]
fn on() -> bool {
    match OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    enabled()
}

/// The two spellings the switch reads: `1/on/true/yes` for on, everything else
/// (including unset) for off.
fn parse(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}

/// Charge one declaration to its arm: `bytes` is the payload the arm states,
/// or `None` for an arm that states a reference instead of bytes.
pub(crate) fn note_declared(arm: PayloadArm, bytes: Option<u64>) {
    if !on() {
        return;
    }
    let (count, total) = ARM_COUNTERS[arm as usize];
    crate::runtime::drain::note_store_route(count);
    if let Some(bytes) = bytes {
        crate::runtime::drain::note_store_route_n(total, bytes);
    }
}

/// Count one statement's texture declarations: the wire's own arm split, the
/// payload those declarations carry, and how much of that payload repeats.
///
/// One call per statement, on the trace the rail is about to state — so the
/// counts are of the bytes a statement *would* carry, and the repeat classes
/// are the two cuts' own upper bounds.
pub(crate) fn note_statement(trace: &ComputeTrace) {
    if !on() {
        return;
    }
    let mut local: std::collections::HashMap<(u64, u64), u32> = std::collections::HashMap::new();
    let mut ordinal: u32 = 0;
    let mut carried: u64 = 0;
    let mut first_n: u64 = 0;
    let mut first_bytes: u64 = 0;
    let mut here_n: u64 = 0;
    let mut here_bytes: u64 = 0;
    let mut earlier_n: u64 = 0;
    let mut earlier_bytes: u64 = 0;
    for texture in textures_of(trace) {
        match &texture.source {
            TextureSource::OwnedBytes(bytes) => {
                let length = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
                carried = carried.saturating_add(length);
                crate::runtime::drain::note_store_route("tex_payload_wire_owned_n");
                crate::runtime::drain::note_store_route_n("tex_payload_wire_owned_bytes", length);
                let key = payload_key(bytes);
                let seen_earlier = boot_seen(&key);
                if local.contains_key(&key) {
                    here_n += 1;
                    here_bytes = here_bytes.saturating_add(length);
                } else if seen_earlier {
                    earlier_n += 1;
                    earlier_bytes = earlier_bytes.saturating_add(length);
                } else {
                    first_n += 1;
                    first_bytes = first_bytes.saturating_add(length);
                }
                local.insert(key, ordinal);
            }
            TextureSource::StagedLease(_) | TextureSource::BorrowedNoCopy(_) => {
                crate::runtime::drain::note_store_route("tex_payload_wire_lease_n");
            }
            TextureSource::TraceView => {
                crate::runtime::drain::note_store_route("tex_payload_wire_trace_view_n");
            }
            TextureSource::PassEntrySnapshot => {
                crate::runtime::drain::note_store_route("tex_payload_wire_snapshot_n");
            }
        }
        ordinal = ordinal.saturating_add(1);
    }
    drop(local);
    crate::runtime::drain::note_store_route("tex_payload_stmt_n");
    crate::runtime::drain::note_store_route_n("tex_payload_stmt_bytes", carried);
    crate::runtime::drain::note_store_route_n("tex_payload_first_n", first_n);
    crate::runtime::drain::note_store_route_n("tex_payload_first_bytes", first_bytes);
    crate::runtime::drain::note_store_route_n("tex_payload_repeat_here_n", here_n);
    crate::runtime::drain::note_store_route_n("tex_payload_repeat_here_bytes", here_bytes);
    crate::runtime::drain::note_store_route_n("tex_payload_repeat_earlier_n", earlier_n);
    crate::runtime::drain::note_store_route_n("tex_payload_repeat_earlier_bytes", earlier_bytes);
}

/// Every texture declaration of one trace, in the order the wire states them.
///
/// The order is the encoder's own (`put_trace` walks the passes, and the
/// texture block of a render entry — or of every draw the multi-draw list
/// carries — is written where the pass states it), which is what makes the
/// ordinal both ends of the wire count agree on.
pub(crate) fn textures_of(trace: &ComputeTrace) -> Vec<&TextureView> {
    let mut textures = Vec::new();
    for pass in &trace.passes {
        match pass {
            TracePass::Compute(pass) => textures.extend(pass.textures.iter()),
            TracePass::Render(pass) => textures.extend(pass.textures.iter()),
            TracePass::RenderDraws(list) => {
                textures.extend(list.head.textures.iter());
                for draw in &list.tail {
                    textures.extend(draw.textures.iter());
                }
            }
            TracePass::Landing(_) => {}
        }
    }
    textures
}

/// The digest and length one payload is filed under.
///
/// The length is half the key rather than folded into the hash, so two
/// payloads of different lengths can never collide into one identity. The
/// digest is FNV-1a 64 — a census reading, not a security boundary — and a
/// caller that *acts* on a hit owes a byte compare of its own (the rewrite arm
/// does exactly that).
fn payload_key(bytes: &[u8]) -> (u64, u64) {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    (hash, u64::try_from(bytes.len()).unwrap_or(u64::MAX))
}

/// The identities the boot has already carried, with the count of the
/// identities that could not be filed.
fn boot_table() -> &'static std::sync::Mutex<std::collections::HashMap<(u64, u64), ()>> {
    static IDENTITIES: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<(u64, u64), ()>>,
    > = std::sync::OnceLock::new();
    IDENTITIES.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Whether an earlier statement carried this identity, filing it when it did
/// not (capped, with the overflow counted by name).
fn boot_seen(key: &(u64, u64)) -> bool {
    let Ok(mut table) = boot_table().lock() else {
        return false;
    };
    if let Some(()) = table.get(key) {
        return true;
    }
    if table.len() >= BOOT_IDS_MAX {
        crate::runtime::drain::note_store_route("tex_payload_boot_ids_overflow_n");
        return false;
    }
    table.insert(*key, ());
    crate::runtime::drain::note_store_route("tex_payload_boot_ids_n");
    false
}

#[cfg(test)]
mod tests {
    use super::{
        note_declared, note_statement, parse, payload_key, set_texture_payload_census_for_test,
        textures_of, PayloadArm,
    };
    use crate::runtime::drain::store_route_count;
    use metal_api_core::provider::{
        AllocationId, CompletionPolicy, ComputePass, ComputeTrace, DeviceEpoch, Dispatch,
        DispatchKind, DispatchType, OperationId, PipelineId, TextureAccess, TextureFormat,
        TextureSource, TextureType, TextureView, TracePass, ViewId, PROVIDER_SCHEMA_VERSION,
    };

    /// The tests that arm the census read process-global counters, so they take
    /// one lock rather than reading each other's charges.
    static CENSUS_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// One sampled texture declaration carrying `bytes`.
    fn texture(view: u64, bytes: &[u8]) -> TextureView {
        TextureView {
            view_id: ViewId::new(view),
            metal_binding: 0,
            allocation_id: AllocationId::new(view),
            texture_type: TextureType::D2,
            format: TextureFormat::Rgba8Unorm,
            width: 1,
            height: u64::try_from(bytes.len() / 4).unwrap_or(1),
            depth: 1,
            array_length: 1,
            sample_count: 1,
            access: TextureAccess::Sampled,
            source: TextureSource::OwnedBytes(bytes.to_vec()),
        }
    }

    /// One compute pass carrying `textures`, over a pipeline id the fixture
    /// names so two passes read apart.
    fn compute_pass(pipeline: u64, textures: Vec<TextureView>) -> TracePass {
        TracePass::Compute(ComputePass {
            pipeline: PipelineId::new(pipeline),
            buffers: Vec::new(),
            textures,
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
        })
    }

    /// A trace whose passes are `passes`.
    fn trace_of(passes: Vec<TracePass>) -> ComputeTrace {
        ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: DeviceEpoch::new(1),
            operation_id: OperationId::new(1),
            pipelines: Vec::new(),
            encoder_dispatch_type: DispatchType::Serial,
            passes,
            completion_policy: CompletionPolicy::HostReadback,
            heap: None,
            indirect: None,
        }
    }

    #[test]
    fn the_switch_reads_the_spellings_a_launcher_writes() {
        for on in ["1", "on", "true", "yes", " on "] {
            assert!(parse(Some(on)), "{on:?} is not read as on");
        }
        for off in ["", "0", "off", "false", "no", "onward"] {
            assert!(!parse(Some(off)), "{off:?} is read as on");
        }
        assert!(!parse(None));
    }

    #[test]
    fn a_payload_is_keyed_by_its_bytes_and_its_length() {
        assert_eq!(payload_key(&[]), payload_key(&[]));
        assert_eq!(payload_key(&[0; 4]), payload_key(&[0; 4]));
        assert_ne!(payload_key(&[0; 4]), payload_key(&[0; 5]));
        assert_ne!(payload_key(&[1; 4]), payload_key(&[2; 4]));
    }

    #[test]
    fn the_walk_states_every_texture_the_wire_states_in_wire_order() {
        // The wire writes a pass's texture block where the pass states it, and
        // the passes in the trace's own order: the walk has to visit the
        // declarations in exactly that order for an ordinal to mean one thing
        // on both ends.
        let trace = trace_of(vec![
            compute_pass(
                1,
                vec![texture(1, &[1, 2, 3, 4]), texture(2, &[5, 6, 7, 8])],
            ),
            compute_pass(2, vec![texture(3, &[9, 10, 11, 12])]),
        ]);
        let views: Vec<u64> = textures_of(&trace)
            .iter()
            .map(|texture| texture.view_id.get())
            .collect();
        assert_eq!(views, vec![1, 2, 3], "the walk's order is the wire's");
    }

    #[test]
    fn the_switch_off_moves_no_counter() {
        let _guard = CENSUS_TEST
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // The counters are process-global and a sibling test may have charged
        // them already, so the reading is the *delta* this call makes: off, it
        // makes none.
        let before = (
            store_route_count("tex_payload_gathered_n"),
            store_route_count("tex_payload_gathered_bytes"),
            store_route_count("tex_payload_window_n"),
            store_route_count("tex_payload_stmt_bytes"),
        );
        set_texture_payload_census_for_test(Some(false));
        note_declared(PayloadArm::Gathered, Some(4096));
        note_declared(PayloadArm::Window, None);
        note_statement(&trace_of(vec![compute_pass(
            1,
            vec![texture(1, &[0x44; 256])],
        )]));
        let after = (
            store_route_count("tex_payload_gathered_n"),
            store_route_count("tex_payload_gathered_bytes"),
            store_route_count("tex_payload_window_n"),
            store_route_count("tex_payload_stmt_bytes"),
        );
        assert_eq!(after, before, "off, no counter moves");
        set_texture_payload_census_for_test(None);
    }

    #[test]
    fn the_arms_count_what_each_one_states_and_what_it_does_not() {
        let _guard = CENSUS_TEST
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        set_texture_payload_census_for_test(Some(true));
        note_declared(PayloadArm::Gathered, Some(4096));
        note_declared(PayloadArm::Frame, Some(1024));
        note_declared(PayloadArm::Window, None);
        note_declared(PayloadArm::Snapshot, None);
        assert_eq!(store_route_count("tex_payload_gathered_n"), 1);
        assert_eq!(store_route_count("tex_payload_gathered_bytes"), 4096);
        assert_eq!(store_route_count("tex_payload_frame_n"), 1);
        assert_eq!(store_route_count("tex_payload_frame_bytes"), 1024);
        // A reference arm states no payload, so its byte counter stays empty:
        // the count is the reading, and a zero beside it would be a second
        // claim about the same fact.
        assert_eq!(store_route_count("tex_payload_window_n"), 1);
        assert_eq!(store_route_count("tex_payload_snapshot_n"), 1);
        assert_eq!(store_route_count("tex_payload_window_bytes"), 0);
        assert_eq!(store_route_count("tex_payload_snapshot_bytes"), 0);
        set_texture_payload_census_for_test(None);
    }

    #[test]
    fn a_statement_reads_its_payload_apart_from_the_bytes_it_repeats() {
        let _guard = CENSUS_TEST
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        set_texture_payload_census_for_test(Some(true));
        let before = (
            store_route_count("tex_payload_stmt_bytes"),
            store_route_count("tex_payload_first_bytes"),
            store_route_count("tex_payload_repeat_here_bytes"),
            store_route_count("tex_payload_repeat_earlier_bytes"),
        );
        // One statement: 4 KiB carried, then the same 4 KiB again — the
        // statement-local class — and 1 KiB that is new to this statement and
        // to the boot.
        let statement = trace_of(vec![
            compute_pass(
                1,
                vec![texture(1, &[0x11; 4096]), texture(2, &[0x22; 1024])],
            ),
            compute_pass(2, vec![texture(3, &[0x11; 4096])]),
        ]);
        note_statement(&statement);
        let after = (
            store_route_count("tex_payload_stmt_bytes"),
            store_route_count("tex_payload_first_bytes"),
            store_route_count("tex_payload_repeat_here_bytes"),
            store_route_count("tex_payload_repeat_earlier_bytes"),
        );
        assert_eq!(after.0 - before.0, 4096 + 1024 + 4096, "bytes carried");
        assert_eq!(after.1 - before.1, 4096 + 1024, "first copies");
        assert_eq!(after.2 - before.2, 4096, "in-statement repeat");
        assert_eq!(after.3 - before.3, 0, "nothing earlier yet");

        // A second statement carrying the same 4 KiB is the *other* class: the
        // bytes an earlier statement already carried, which is the class that
        // owes a key and a generation rather than a statement-local ordinal.
        let before = (
            store_route_count("tex_payload_first_bytes"),
            store_route_count("tex_payload_repeat_earlier_bytes"),
        );
        note_statement(&trace_of(vec![compute_pass(
            3,
            vec![texture(4, &[0x11; 4096])],
        )]));
        assert_eq!(store_route_count("tex_payload_first_bytes") - before.0, 0);
        assert_eq!(
            store_route_count("tex_payload_repeat_earlier_bytes") - before.1,
            4096
        );
        set_texture_payload_census_for_test(None);
    }

    #[test]
    fn a_reference_arm_is_counted_by_the_wire_split_and_not_as_payload() {
        let _guard = CENSUS_TEST
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        set_texture_payload_census_for_test(Some(true));
        let before = (
            store_route_count("tex_payload_wire_owned_n"),
            store_route_count("tex_payload_wire_trace_view_n"),
            store_route_count("tex_payload_wire_snapshot_n"),
            store_route_count("tex_payload_wire_lease_n"),
            store_route_count("tex_payload_stmt_bytes"),
        );
        let mut produced = texture(5, &[]);
        produced.source = TextureSource::TraceView;
        let mut snapshot = texture(6, &[]);
        snapshot.source = TextureSource::PassEntrySnapshot;
        let mut leased = texture(7, &[]);
        leased.source = TextureSource::StagedLease(metal_api_core::provider::LeaseId::new(9));
        note_statement(&trace_of(vec![compute_pass(
            4,
            vec![texture(8, &[0x33; 256]), produced, snapshot, leased],
        )]));
        assert_eq!(store_route_count("tex_payload_wire_owned_n") - before.0, 1);
        assert_eq!(
            store_route_count("tex_payload_wire_trace_view_n") - before.1,
            1
        );
        assert_eq!(
            store_route_count("tex_payload_wire_snapshot_n") - before.2,
            1
        );
        assert_eq!(store_route_count("tex_payload_wire_lease_n") - before.3, 1);
        assert_eq!(
            store_route_count("tex_payload_stmt_bytes") - before.4,
            256,
            "a reference arm carries no payload"
        );
        set_texture_payload_census_for_test(None);
    }
}
