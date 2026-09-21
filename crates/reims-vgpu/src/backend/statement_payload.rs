//! The owner's half of the statement payload table (statement economy W4, task
//! E-SW3; `metal_api_core::statement_payload`).
//!
//! The census this increment starts from
//! (`.agents/tasks/root/e_statement_texture_payload-report.md`) measured the
//! sampled textures' payload at 608 KB per *crossing* statement — and what it
//! is: over one production-pose round, 265 distinct payloads carried by a whole
//! boot, 98% of the delivered bytes a re-send of bytes an earlier statement
//! carried, and 0.0003% a repeat of bytes the same statement carried. So the
//! bytes come off the wire by *naming* them, which needs a table the provider
//! keeps across statements and a ledger the sender keeps beside it.
//!
//! # The sender is the only end that decides
//!
//! [`plan`] is called once per statement, on the composed trace, before the
//! frame is encoded — and only for the statements that will actually cross the
//! wire (a trace with no lease at all keeps the in-process path, where there is
//! no wire to take bytes off, so it is left exactly as it was). Every
//! byte-carrying declaration is looked up in the ledger:
//!
//! - the ledger holds these bytes ⇒ the statement names them
//!   (`TextureSource::SlottedBytes`: slot, length, digest) and carries none;
//! - it does not ⇒ the statement declares them into a slot the ledger plans
//!   (`TextureSource::OwnedInSlot`) or, when the payload does not fit the
//!   table's bounds, carries them exactly as every round before this one did.
//!
//! What the statement decided travels *in the statement*, so the provider's
//! table is a deterministic function of the same payload its own reader
//! resolved: the two ends agree after every statement both processed, and the
//! reference arm's digest is what turns a desync into a named refusal instead
//! of another payload's bytes.
//!
//! # What is owed, and when
//!
//! A plan is not a filing: [`PlannedStatement`] carries the declarations the
//! statement stated, and it is committed only once the provider's reader has
//! resolved the frame ([`VulkanComputeProvider::resolve_statement_payloads`]).
//! A statement that never got that far — a frame the encoder refused, a wire
//! hop that never happened — drops its plan, so neither end holds an entry the
//! other does not.

use std::sync::{Mutex, OnceLock};

use metal_api_core::provider::{ComputeTrace, TextureSource};
use metal_api_core::statement_payload::{payload_digest, PayloadLedger, PayloadPlan};

/// The switch a launcher writes to arm the mechanism. **Back to off by
/// default** (2026-09-22): census v65 flipped it on and read the red line
/// `draws_skipped_after_engine_refusal = 3 468`, every skip
/// `refused_by=trace_admission`, behind seven `statement_payload_table_full`
/// refusals (six of them the per-pipeline refusal line, which dedupes on
/// `(pipeline, slug)`, and one that came back through the batch path). A
/// refusal kills the whole statement rather than carrying that one
/// declaration's bytes, so six statements were refused once each and re-walked
/// for the rest of the round.
///
/// # What the two ends disagreed about, and what the E tip did about it
///
/// Not *which* bound: the sender's ledger read one of its two bounds and the
/// provider's `declare` read both. A plan's fresh-slot branch returned a slot
/// number below `PAYLOAD_TABLE_SLOTS` without asking `payload_fits`, on the
/// reading that a free slot number is room — and the byte bound is the one that
/// binds first by orders of magnitude. v65's own counters say which:
/// `stmt_payload_declared_*` reached **531 265 024 B in 100 filings** against a
/// 512 MiB bound, `stmt_payload_carried_n` stayed **0** for the whole round (a
/// plan that read the byte bound would have replaced an entry or carried), and
/// the refusals start at the first payload the remaining 5.6 MB could not hold.
///
/// `metal_api_core::statement_payload::PayloadLedger::plan` (E
/// `feat-payload-bound`, `5ee8ba5`) now reads `payload_fits` on **every**
/// branch: a fresh slot is room only while the whole budget holds it, past that
/// the plan replaces the least recently used entry into that entry's own slot,
/// and a payload even that cannot hold is carried — the policy the module docs
/// already claimed. The two-ends walk that fails on the old plan and passes on
/// this one is in that module's own tests
/// (`the_two_ends_hold_the_same_table_over_the_shape_census_v65_walked`), and
/// the provider's own caps are a wiring guard again rather than a second policy.
///
/// The arm stays here until a census reads the red line at 0 on a pose that
/// crosses the bound: E-SW3's 120 s arms saw ~500 MB of distinct payloads and
/// never reached it, which is why the 430 s round is the one that caught it.
/// The cut is priced (616 488 → 22 285 B a statement). unset is off,
/// `1/on/true/yes` arms it.
const SWITCH: &str = "REIMS_VGPU_STATEMENT_PAYLOAD_TABLE";

/// Whether this process states the payload table's arms.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var(SWITCH).ok().as_deref().map(str::trim),
            Some("1") | Some("on") | Some("true") | Some("yes")
        )
    })
}

/// Arm or disarm the mechanism in a test, leaving the environment's own answer
/// alone (`zero_fill_decl_arm`'s shape).
static OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Tests only; the runtime never calls it.
#[doc(hidden)]
pub fn set_statement_payload_table_for_test(arm: Option<bool>) {
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

#[inline]
fn on() -> bool {
    match OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    enabled()
}

/// The ledger this device's statements plan against: one table per process,
/// scoped by the device epoch each statement states.
fn ledger() -> &'static Mutex<PayloadLedger> {
    static LEDGER: OnceLock<Mutex<PayloadLedger>> = OnceLock::new();
    LEDGER.get_or_init(|| Mutex::new(PayloadLedger::new()))
}

/// What one statement's plan did, and what it owes the ledger.
#[derive(Debug, Default)]
pub(crate) struct PlannedStatement {
    /// The `(slot, digest, length)` each declaration arm filed, in the order
    /// the statement states them.
    declarations: Vec<(u32, u128, u64)>,
    /// Declarations that stated the table's own arm.
    pub(crate) declared_n: u64,
    /// Payload bytes those declarations still carried.
    pub(crate) declared_bytes: u64,
    /// Declarations that named bytes an earlier statement filed.
    pub(crate) referenced_n: u64,
    /// The bytes those references named — the payload this statement did not
    /// carry.
    pub(crate) referenced_bytes: u64,
    /// Declarations the table's bounds left on the ordinary arm.
    pub(crate) carried_n: u64,
    /// Their bytes.
    pub(crate) carried_bytes: u64,
}

impl PlannedStatement {
    /// Whether the statement stated any of the table's arms.
    pub(crate) fn any(&self) -> bool {
        self.declared_n > 0 || self.referenced_n > 0
    }

    /// Take the statement's declarations into the ledger: the point after
    /// which the two ends hold the same table. Called once the provider's
    /// reader has resolved the frame.
    pub(crate) fn commit(&mut self) {
        let declarations = std::mem::take(&mut self.declarations);
        let Ok(mut ledger) = ledger().lock() else {
            return;
        };
        for (slot, digest, length) in declarations {
            ledger.commit(slot, digest, length);
        }
        // The mechanism's own reading, in the census's own window: what the
        // statement filed, what it named instead of carrying, and what the
        // table's bounds left on the ordinary arm.
        crate::runtime::drain::note_store_route_n("stmt_payload_declared_n", self.declared_n);
        crate::runtime::drain::note_store_route_n(
            "stmt_payload_declared_bytes",
            self.declared_bytes,
        );
        crate::runtime::drain::note_store_route_n("stmt_payload_referenced_n", self.referenced_n);
        crate::runtime::drain::note_store_route_n(
            "stmt_payload_referenced_bytes",
            self.referenced_bytes,
        );
        crate::runtime::drain::note_store_route_n("stmt_payload_carried_n", self.carried_n);
        crate::runtime::drain::note_store_route_n("stmt_payload_carried_bytes", self.carried_bytes);
    }
}

/// Plan one statement: every byte-carrying declaration of `trace` that the
/// ledger can name becomes the reference arm, every one it can file becomes a
/// declaration arm, and the rest keep the ordinary arm exactly as they were.
///
/// Off, nothing is walked and nothing on the trace changes.
pub(crate) fn plan(trace: &mut ComputeTrace) -> PlannedStatement {
    let mut planned = PlannedStatement::default();
    if !on() {
        return planned;
    }
    let Ok(mut ledger) = ledger().lock() else {
        return planned;
    };
    ledger.scope(trace.device_epoch);
    for texture in trace.texture_declarations_mut() {
        let taken = std::mem::replace(&mut texture.source, TextureSource::PassEntrySnapshot);
        match taken {
            TextureSource::OwnedBytes(bytes) => {
                let length = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
                let digest = payload_digest(&bytes);
                match ledger.plan(digest, length) {
                    PayloadPlan::Reuse(slot) => {
                        planned.referenced_n += 1;
                        planned.referenced_bytes = planned.referenced_bytes.saturating_add(length);
                        texture.source = TextureSource::SlottedBytes {
                            slot,
                            length,
                            digest,
                        };
                    }
                    PayloadPlan::Declare(slot) => {
                        planned.declared_n += 1;
                        planned.declared_bytes = planned.declared_bytes.saturating_add(length);
                        planned.declarations.push((slot, digest, length));
                        texture.source = TextureSource::OwnedInSlot { slot, bytes };
                    }
                    PayloadPlan::Carry => {
                        planned.carried_n += 1;
                        planned.carried_bytes = planned.carried_bytes.saturating_add(length);
                        texture.source = TextureSource::OwnedBytes(bytes);
                    }
                }
            }
            other => texture.source = other,
        }
    }
    planned
}

#[cfg(test)]
mod tests {
    use super::{plan, set_statement_payload_table_for_test};
    use metal_api_core::provider::{
        AllocationId, CompletionPolicy, ComputePass, ComputeTrace, DeviceEpoch, Dispatch,
        DispatchKind, DispatchType, OperationId, PipelineId, TextureAccess, TextureFormat,
        TextureSource, TextureType, TextureView, TracePass, ViewId, PROVIDER_SCHEMA_VERSION,
    };

    fn texture(view: u64, bytes: &[u8]) -> TextureView {
        TextureView {
            view_id: ViewId::new(view),
            metal_binding: 0,
            allocation_id: AllocationId::new(view),
            texture_type: TextureType::D2,
            format: TextureFormat::R8Unorm,
            width: u64::try_from(bytes.len()).unwrap_or(1),
            height: 1,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            access: TextureAccess::Sampled,
            source: TextureSource::OwnedBytes(bytes.to_vec()),
        }
    }

    fn trace(textures: Vec<TextureView>) -> ComputeTrace {
        ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: DeviceEpoch::new(1),
            operation_id: OperationId::new(1),
            pipelines: Vec::new(),
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![TracePass::Compute(ComputePass {
                pipeline: PipelineId::new(1),
                buffers: Vec::new(),
                textures,
                dispatch: Dispatch {
                    kind: DispatchKind::ThreadsExact,
                    grid: [1, 1, 1],
                    threads_per_threadgroup: [1, 1, 1],
                },
            })],
            completion_policy: CompletionPolicy::HostReadback,
            heap: None,
            indirect: None,
        }
    }

    /// What one declaration's arm came to, as the test reads it.
    fn arm(trace: &ComputeTrace) -> String {
        match &trace.passes[0].as_compute().expect("compute pass").textures[0].source {
            TextureSource::OwnedBytes(bytes) => format!("owned:{}", bytes.len()),
            TextureSource::OwnedInSlot { slot, bytes } => {
                format!("declared:{}:{}", slot, bytes.len())
            }
            TextureSource::SlottedBytes { slot, .. } => format!("referenced:{slot}"),
            other => format!("other:{other:?}"),
        }
    }

    #[test]
    fn off_changes_nothing_and_plans_nothing() {
        set_statement_payload_table_for_test(Some(false));
        let mut trace = trace(vec![texture(1, &[0x11; 64])]);
        let planned = plan(&mut trace);
        assert!(!planned.any(), "off, nothing is planned");
        assert_eq!(
            arm(&trace),
            "owned:64",
            "off, the statement keeps its bytes"
        );
        set_statement_payload_table_for_test(None);
    }

    #[test]
    fn a_statement_declares_what_the_next_one_references() {
        set_statement_payload_table_for_test(Some(true));
        // A payload nothing has carried yet is declared into a slot…
        let mut first = trace(vec![texture(1, &[0x22; 4096])]);
        let mut planned = plan(&mut first);
        assert_eq!(planned.declared_n, 1);
        assert_eq!(planned.declared_bytes, 4096);
        assert_eq!(planned.referenced_n, 0);
        assert_eq!(planned.declared_n, 1, "the plan owes the ledger a filing");
        let declared = arm(&first);
        assert!(declared.starts_with("declared:"), "{declared}");
        // …and the ledger only holds it once the frame resolved on the other
        // end.
        let mut second = trace(vec![texture(2, &[0x22; 4096])]);
        assert_eq!(arm(&second), "owned:4096");
        assert_eq!(plan(&mut second).referenced_n, 0);
        planned.commit();
        let mut third = trace(vec![texture(3, &[0x22; 4096])]);
        let planned = plan(&mut third);
        assert_eq!(planned.referenced_n, 1);
        assert_eq!(planned.referenced_bytes, 4096);
        assert_eq!(planned.declared_n, 0, "the second statement names it");
        assert!(arm(&third).starts_with("referenced:"), "{}", arm(&third));
        // Other bytes are their own payload, not a reference.
        let mut other = trace(vec![texture(4, &[0x23; 4096])]);
        assert_eq!(plan(&mut other).declared_n, 1);
        // And a plan nobody committed leaves the ledger where it was.
        let mut fourth = trace(vec![texture(5, &[0x24; 16])]);
        let dropped = plan(&mut fourth);
        assert_eq!(dropped.declared_n, 1, "the plan owes the ledger a filing");
        drop(dropped);
        let mut fifth = trace(vec![texture(6, &[0x24; 16])]);
        assert_eq!(
            plan(&mut fifth).declared_n,
            1,
            "a plan that never crossed is declared again rather than named"
        );
        set_statement_payload_table_for_test(None);
    }
}
