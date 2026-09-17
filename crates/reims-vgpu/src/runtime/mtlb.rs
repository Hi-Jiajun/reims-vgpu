//! Guest function objects: loading the MTLB container, and carving the
//! wrapped AIR out of it.
//!
//! Each function object's descriptor names an MTLB container in guest memory;
//! metal2vulkan consumes the LLVM BitcodeWrapper (`0x0b17c0de`) record inside.
//! [`load_mtlb`](crate::runtime::mtlb::load_mtlb) does the first half
//! (object list → container bytes) and
//! [`extract_air`](crate::runtime::mtlb::extract_air) the second. Port of archive `reims-vgpu-backend-vulkan`
//! `mtlb.rs` (structural carve only — no guest scan).
//!
//! # One wrapper is the only carve that answers "the function I asked for"
//!
//! `extract_air` carves the *one* BitcodeWrapper a container holds and refuses
//! any container that holds several. The refusal is the fix for a silent
//! miscompile: a `.metallib` linked from two `.metal` files is one MTLB whose
//! directory lists two functions, each with its own wrapper — `mesh_main` and
//! `object_main` in `tests/fixtures/icb_object_buf.metallib`, say — and this
//! function used to return whichever wrapper came first. A request for the
//! second function then translated, cached and bound the **first** function's
//! AIR, which is a wrong shader rather than a missing one.
//!
//! Nothing on this side of the guest identifies the function that was asked
//! for, so "pick the right one" is not on the table: the function descriptor
//! carries `blob_gva` / `blob_size` and a `function_id` no consumer reads
//! (`runtime/decode/resource/mod.rs:1657-1661`, `:2449-2466`), and the container
//! carries names with no request to match them against. What *is* decidable is
//! whether the container is ambiguous, and that needs no format parsing at all:
//! a wrapper count is a count of candidate functions.
//!
//! # One loader, two rails
//!
//! The compute rail and the draw rail both need a function's container, and
//! until this module took the loader they each had their own copy of it — the
//! same six steps in the same order, down to a verbatim-shared comment about the
//! guest's `blob_size` being authoritative. They had drifted in the way this
//! project's twin functions always drift: the compute copy named all six of its
//! failures in the fail log and the draw copy returned a bare `None` from every
//! one of them, so a draw that lost its shader said only `MissingMtlb` and never
//! which of six things went wrong. That is why the loader takes an
//! [`AirLoadRail`](crate::runtime::mtlb::AirLoadRail) rather than living in
//! either caller.
//!
//! # What the draw rail's new lines cost, measured
//!
//! Giving the per-frame rail six fail lines it never had is a volume question,
//! so it was measured rather than argued: a driven x86/Vulkan boot (Safari drag,
//! 2 685 posted events, ~35 Hz median present) ran **177 746 draws**, hence
//! ~355 000 calls to this loader — two per draw, vertex and fragment — and
//! emitted **zero** `draw_load_mtlb` lines.
//!
//! That zero is a healthy one rather than an unarmed detector, and the thing
//! that says so is independent of this module: the coarse `MissingMtlb`-class
//! declines its callers raise were *also* zero on that boot, and they predate
//! the emission. The loader was not failing quietly before; it was not failing.
//! So a `draw_load_mtlb` line is a real event, and the flood this could have
//! been does not exist on a healthy guest.

use crate::model::DeviceState;
use crate::runtime::decode::resource::{decode_function_descriptor, OBJECT_TYPE_FUNCTION};
use crate::runtime::draw::host_alloc_len;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::{gva_mem, objects};

/// LLVM BitcodeWrapperHeader magic `0x0b17c0de` LE.
///
/// Public because [`extract_air`] is, and everything that function does happens
/// *after* it finds this: a caller that wants to reach the wrapper-header
/// arithmetic must be able to name the magic rather than write the four bytes
/// out a second time.
pub const AIR_WRAP_MAGIC: [u8; 4] = [0xde, 0xc0, 0x17, 0x0b];
const WRAPPER_HEADER_LEN: usize = 0x14;

/// A structural refusal while locating the LLVM BitcodeWrapper inside an MTLB.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MtlbDecline {
    WrappedAirMissing {
        data_len: usize,
    },
    WrapperHeaderTruncated {
        offset: usize,
        data_len: usize,
    },
    BlobOutOfBounds {
        offset: usize,
        blob_len: u64,
        data_len: usize,
    },
    /// The container holds more than one wrapped AIR function, and nothing in
    /// the request says which one the guest asked for.
    ///
    /// A refusal rather than a guess: returning the first wrapper translated
    /// the wrong function, silently and for the whole life of the boot.
    FunctionSelectionUnsupported {
        wrappers: usize,
        dir_end: u64,
        data_len: usize,
    },
}

impl crate::observe::Decline for MtlbDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::WrappedAirMissing { .. } => "mtlb_wrapped_air_missing",
            Self::WrapperHeaderTruncated { .. } => "mtlb_wrapper_header_truncated",
            Self::BlobOutOfBounds { .. } => "mtlb_blob_out_of_bounds",
            Self::FunctionSelectionUnsupported { .. } => "mtlb_function_selection_unsupported",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::WrappedAirMissing { data_len } => vec![("data_len", data_len.to_string())],
            Self::WrapperHeaderTruncated { offset, data_len } => vec![
                ("offset", offset.to_string()),
                ("data_len", data_len.to_string()),
            ],
            Self::BlobOutOfBounds {
                offset,
                blob_len,
                data_len,
            } => vec![
                ("offset", offset.to_string()),
                ("blob_len", blob_len.to_string()),
                ("data_len", data_len.to_string()),
            ],
            Self::FunctionSelectionUnsupported {
                wrappers,
                dir_end,
                data_len,
            } => vec![
                ("wrappers", wrappers.to_string()),
                ("dir_end", format!("{dir_end:#x}")),
                ("data_len", data_len.to_string()),
            ],
        }
    }
}

crate::observe::decline_display!(MtlbDecline);

impl std::error::Error for MtlbDecline {}

/// Which rail asked for a function's MTLB container.
///
/// The only thing that differs between the two is the event name the failure
/// lines carry. The reason slugs are deliberately bare — `ladder_slug!("", …)`
/// and friends — because the event name already says which load it was, which
/// is the convention [`crate::observe::ladder_slug`] documents; so the event
/// name is the one thing that has to travel in from the caller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AirLoadRail {
    /// Compute dispatch, its sessions, and compute ICB bodies.
    Compute,
    /// Render draws on either backend, and render ICB bodies.
    Draw,
}

impl AirLoadRail {
    /// The fail-log event name for this rail's load failures.
    fn event(self) -> &'static str {
        match self {
            Self::Compute => "compute_load_mtlb",
            Self::Draw => "draw_load_mtlb",
        }
    }
}

/// Load a function object's MTLB container out of guest memory.
///
/// `None` means the caller gets no shader; every reason for it but one is
/// written to the fail log under [`AirLoadRail::event`], because the callers all
/// collapse this into one coarse `MissingMtlb`-class decline and the reason is
/// the only thing that says which of six steps refused.
///
/// The exception is `func_ref == 0`, which is "no function bound" — a legitimate
/// state (a pipeline with no fragment stage, say) that `AGENTS.md` names as a
/// thing not to log. It stays silent.
pub fn load_mtlb<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    func_ref: u32,
    rail: AirLoadRail,
) -> Option<Vec<u8>> {
    if func_ref == 0 {
        return None;
    }
    let report = crate::observe::RungReport::new(rail.event(), "func_ref");
    let miss = |reason: &str, detail: String| -> Option<Vec<u8>> {
        report.reason(task_id, func_ref, reason, &detail);
        None
    };
    let (_entry, desc) = match objects::resolve_descriptor(
        state,
        host,
        task_id,
        func_ref,
        &[OBJECT_TYPE_FUNCTION],
    ) {
        Ok(found) => found,
        Err(rung) => {
            report.rung(task_id, func_ref, rung);
            return None;
        }
    };
    let Ok(f) = decode_function_descriptor(&desc) else {
        return miss(
            crate::observe::ladder_slug!("", desc_decode),
            format!("desc_len={}", desc.len()),
        );
    };
    // The slot holds a function, which is all the model needs to hold its name
    // — and the guest's `CmdDeleteObject` for it arrives after the slot has
    // been cleared, so this is the only moment the name can be taken. Before
    // the blob checks below rather than after: a function whose blob this
    // device cannot read is still a function the guest created and will delete.
    objects::note_named_at_construction(
        state,
        host,
        task_id,
        func_ref,
        "function_model_named",
        "function_model_unnamed",
    );
    if f.blob_gva == 0 || f.blob_size < 4 {
        return miss(
            "bad_blob",
            format!("blob_gva={:#x} blob_size={}", f.blob_gva, f.blob_size),
        );
    }
    // Guest blob_size is authoritative — no product 1 MiB MTLB ceiling.
    let Some(len) = host_alloc_len(f.blob_size as u64) else {
        return miss(
            "host_len",
            format!("blob_gva={:#x} blob_size={}", f.blob_gva, f.blob_size),
        );
    };
    let mut mtlb = vec![0u8; len];
    // Device page_shift (x86=12, arm64=14); the unshifted helper defaults to
    // arm and fails every load on the other geometry.
    if gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        f.blob_gva,
        &mut mtlb,
        state.page_shift,
    )
    .is_err()
    {
        return miss(
            "gva_read",
            format!("blob_gva={:#x} blob_size={}", f.blob_gva, f.blob_size),
        );
    }
    Some(mtlb)
}

/// Extract the wrapped-AIR blob from an MTLB container or bare wrapper.
///
/// Exactly one complete wrapper in `data` is the success case — a bare wrapper
/// (the input shape the offline carver and the provider tests use) and a
/// single-function `.metallib` (what the guest compiles from one `.metal` file)
/// are both that. A container holding several wrappers is
/// [`MtlbDecline::FunctionSelectionUnsupported`], never "the first one": see
/// this module's header for why selection cannot be answered on this side.
///
/// A wrapper candidate is complete when its four magic bytes sit inside `data`
/// and its own `BitcodeOffset + BitcodeSize` fits after them. A candidate whose
/// header is truncated is not counted — which keeps the old
/// [`MtlbDecline::WrapperHeaderTruncated`] — and a stray magic inside bitcode
/// declines the read rather than being carved.
pub fn extract_air(data: &[u8]) -> Result<&[u8], MtlbDecline> {
    let containers = count_wrappers(data);
    match containers
        .decision()
        .ok_or_else(|| containers.decline(data))?
    {
        Decision::Single(off) => blob_at(data, off),
    }
}

/// What the wrapper count alone decides.
#[derive(Debug)]
enum Decision {
    /// The one wrapper every request to this container has to mean.
    Single(usize),
}

/// Where a complete wrapper ends, and the header offset of the first candidate
/// whose header did not fit — the value the old first-match scan would have
/// refused with.
#[derive(Debug, Default)]
struct Wrappers {
    /// Complete wrappers, the refusal's whole input.
    wrappers: usize,
    /// Offset of the first complete wrapper; start of the carve.
    first_complete: Option<usize>,
    first_truncated: Option<usize>,
    /// The container's own count, or `None` for a bare wrapper and for a header
    /// that cannot be read.
    declared_functions: Option<u64>,
}

impl Wrappers {
    /// The carve, when the count leaves exactly one candidate.
    fn decision(&self) -> Option<Decision> {
        match (self.first_complete, self.declared_functions) {
            // The container declares one function and holds one wrapper.
            (Some(off), Some(1)) if self.wrappers == 1 => Some(Decision::Single(off)),
            // Not a container (bare wrapper) and one wrapper.
            (Some(off), None) if self.wrappers == 1 => Some(Decision::Single(off)),
            _ => None,
        }
    }

    /// The refusal for a count that did not decide — by wrapper count where
    /// there are several, and by the old scan's own reasons where there are
    /// none.
    fn decline(&self, data: &[u8]) -> MtlbDecline {
        self.first_complete.map_or_else(
            || {
                self.first_truncated.map_or(
                    MtlbDecline::WrappedAirMissing {
                        data_len: data.len(),
                    },
                    |offset| MtlbDecline::WrapperHeaderTruncated {
                        offset,
                        data_len: data.len(),
                    },
                )
            },
            |_| MtlbDecline::FunctionSelectionUnsupported {
                // A container cannot hold fewer complete wrappers than it
                // declares functions, so the number to report is the larger of
                // the two: the container that *lost* a function must not look
                // like the container that has one.
                wrappers: self
                    .wrappers
                    .max(self.declared_functions.unwrap_or(0) as usize),
                dir_end: directory_end(data),
                data_len: data.len(),
            },
        )
    }
}

/// Count the complete wrappers in `data`, keeping where the first one starts.
fn count_wrappers(data: &[u8]) -> Wrappers {
    let mut found = Wrappers {
        declared_functions: declared_function_count(data),
        ..Wrappers::default()
    };
    let mut from = 0usize;
    while let Some(off) = find_wrap_magic(data, from) {
        from = off + 1;
        match wrapper_at(data, off) {
            Some(_) => {
                found.wrappers += 1;
                if found.first_complete.is_none() {
                    found.first_complete = Some(off);
                }
            }
            None => {
                if found.first_truncated.is_none() {
                    found.first_truncated = Some(off);
                }
            }
        }
    }
    found
}

/// The wrapper's extent by its own header, or `None` when the header is
/// truncated or its declared blob does not fit inside `data`.
fn wrapper_at(data: &[u8], off: usize) -> Option<usize> {
    let header_end = off.checked_add(WRAPPER_HEADER_LEN)?;
    let header = data.get(off..header_end)?;
    let bc_off = u32::from_le_bytes(header[8..12].try_into().ok()?);
    let bc_size = u32::from_le_bytes(header[12..16].try_into().ok()?);
    let blob_len = u64::from(bc_off) + u64::from(bc_size);
    if blob_len < WRAPPER_HEADER_LEN as u64 {
        return None;
    }
    let end = usize::try_from(blob_len)
        .ok()
        .and_then(|len| off.checked_add(len))?;
    if end > data.len() {
        return None;
    }
    Some(end)
}

/// The `MTLB` header's function count (`0x58`), or `None` when `data` is not a
/// container whose spans this decoder can read.
fn declared_function_count(data: &[u8]) -> Option<u64> {
    if !data.starts_with(b"MTLB") {
        return None;
    }
    // The three spans the two callers below read, each checked against the
    // length `data` actually has: 0x10 (`total`), 0x48 (`dir_end`) and 0x58
    // (`function_count`).
    let total = u64::from_le_bytes(data.get(0x10..0x18)?.try_into().ok()?);
    let dir_end = u64::from_le_bytes(data.get(0x48..0x50)?.try_into().ok()?);
    let functions = u64::from(u32::from_le_bytes(data.get(0x58..0x5c)?.try_into().ok()?));
    if total != data.len() as u64 || dir_end > total || functions == 0 {
        return None;
    }
    Some(functions)
}

/// The MTLB directory end (header field at offset `0x48`), reported on the
/// multi-function refusal because it is the boundary the wrappers follow — and
/// `0` when this is not an `MTLB` container, which a bare wrapper is not.
fn directory_end(data: &[u8]) -> u64 {
    data.get(0x48..0x50)
        .filter(|_| data.starts_with(b"MTLB"))
        .map_or(0, |field| {
            u64::from_le_bytes(field.try_into().expect("0x48..0x50 is eight bytes"))
        })
}

fn find_wrap_magic(data: &[u8], from: usize) -> Option<usize> {
    if data.len() < WRAPPER_HEADER_LEN {
        return None;
    }
    (from..=data.len() - AIR_WRAP_MAGIC.len()).find(|&i| data[i..i + 4] == AIR_WRAP_MAGIC)
}

fn blob_at(data: &[u8], off: usize) -> Result<&[u8], MtlbDecline> {
    let header_end = off.saturating_add(WRAPPER_HEADER_LEN);
    if header_end > data.len() {
        return Err(MtlbDecline::WrapperHeaderTruncated {
            offset: off,
            data_len: data.len(),
        });
    }
    let bc_off = u32::from_le_bytes(data[off + 8..off + 12].try_into().unwrap());
    let bc_size = u32::from_le_bytes(data[off + 12..off + 16].try_into().unwrap());
    let blob_len = u64::from(bc_off) + u64::from(bc_size);
    let blob_end = usize::try_from(blob_len)
        .ok()
        .and_then(|len| off.checked_add(len));
    // Guest/header sizes are authoritative — no product MiB ceiling. Only require
    // the declared blob fits inside the MTLB buffer we already loaded.
    if blob_len < WRAPPER_HEADER_LEN as u64 || blob_end.is_none_or(|end| end > data.len()) {
        return Err(MtlbDecline::BlobOutOfBounds {
            offset: off,
            blob_len,
            data_len: data.len(),
        });
    }
    Ok(&data[off..blob_end.expect("bounds checked above")])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
    use crate::protocol::endian::{st32, st64};
    use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::runtime::host::FakeHost;

    /// Task 1 with a one-entry object list whose ref 1 holds `object_type`, and
    /// a descriptor blob of `desc` at GVA 0x40.
    fn task_with_one_object(
        host: &mut FakeHost,
        state: &mut DeviceState,
        object_type: u8,
        desc: &[u8],
    ) {
        let dir_gpa = 2u64 << PAGE_SHIFT_ARM64E;
        let root_gpa = 3u64 << PAGE_SHIFT_ARM64E;
        let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x4000, 0);
        host.map_range(data_gpa, 0x200, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root_gpa, &d[..4]);

        state.define_task(1, 0x1000, 2);
        assert!(state.set_object_list(1, 0, 8));
        let mut entry = [0u8; 12];
        st32(
            &mut entry[0..],
            u32::from(object_type) | ((desc.len() as u32) << 8),
        );
        st64(&mut entry[4..12], 0x40);
        let _ = host.write_gpa(data_gpa + 12, &entry);
        let _ = host.write_gpa(data_gpa + 0x40, desc);
    }

    /// Both rails name the same refusal, each under its own event.
    ///
    /// The draw rail is the half this pins: it used to be a separate copy of
    /// this loader that returned a bare `None` from all six of its failure
    /// points, so a draw whose shader would not load reported `MissingMtlb` and
    /// nothing about which step refused. The compute half is here because the
    /// two events must stay distinguishable — one shared loader emitting one
    /// event name would make the fail log unable to say which rail lost work.
    #[test]
    fn both_rails_name_the_rung_that_refused_under_their_own_event() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        // Object mapper-ref-texture is an IOSurface, not the function this loads.
        task_with_one_object(&mut host, &mut state, 11, &[0u8; 0x20]);

        let cap = crate::observe::FailCapture::start();
        assert!(load_mtlb(&state, &host, 1, 1, AirLoadRail::Draw).is_none());
        let line = cap.one("draw_load_mtlb");
        assert!(
            line.contains(&format!(
                "reason={}",
                crate::observe::ladder_slug!("", wrong_type)
            )) && line.contains("ot=11"),
            "the draw rail must name the rung and the tag it found: {line}"
        );
        drop(cap);

        let cap = crate::observe::FailCapture::start();
        assert!(load_mtlb(&state, &host, 1, 1, AirLoadRail::Compute).is_none());
        assert!(
            cap.one("compute_load_mtlb").contains("ot=11"),
            "the compute rail keeps its own event name"
        );
    }

    /// `func_ref == 0` is the one refusal that stays silent, because it is not
    /// one: a pipeline with no fragment stage binds no fragment function.
    #[test]
    fn an_unbound_function_ref_says_nothing() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        task_with_one_object(&mut host, &mut state, OBJECT_TYPE_FUNCTION, &[0u8; 0x20]);

        for rail in [AirLoadRail::Draw, AirLoadRail::Compute] {
            let cap = crate::observe::FailCapture::start();
            assert!(load_mtlb(&state, &host, 1, 0, rail).is_none());
            assert!(
                cap.lines().is_empty(),
                "an unbound ref must spend no line on {rail:?}: {:?}",
                cap.lines()
            );
        }
    }

    #[test]
    fn extract_bare_wrapper() {
        // Minimal synthetic: magic + version + offset 0x14 + size 4 + cpu + 4 body bytes.
        let mut data = vec![0u8; 0x18];
        data[0..4].copy_from_slice(&AIR_WRAP_MAGIC);
        data[4..8].copy_from_slice(&0u32.to_le_bytes()); // version
        data[8..12].copy_from_slice(&0x14u32.to_le_bytes()); // BitcodeOffset
        data[12..16].copy_from_slice(&4u32.to_le_bytes()); // BitcodeSize
        data[0x14..0x18].copy_from_slice(&[1, 2, 3, 4]);
        let air = extract_air(&data).expect("air");
        assert_eq!(air.len(), 0x18);
    }

    /// The reference the refusal is measured against: a container with one
    /// function must carve exactly what the first-magic scan carved, first
    /// magic as well as last byte.
    #[test]
    fn a_single_function_container_carves_the_same_bytes_the_scan_carved() {
        let container = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/compute_mul3add1.mtlb"
        ))
        .expect("compute_mul3add1.mtlb fixture");
        let carved = extract_air(&container).expect("one function carves");
        let scan = find_wrap_magic(&container, 0).expect("the fixture holds a wrapper");
        assert_eq!(carved, blob_at(&container, scan).expect("scan carves"));
        assert_eq!(carved.len(), 0x0a90);
    }

    /// A wrapper header that starts four bytes before the end of the data.
    ///
    /// The old scan found the magic and refused with a truncated header; the
    /// per-candidate scan has to keep that answer for a `data` that ends this
    /// way — and must not treat the short candidate as a wrapper it counted.
    #[test]
    fn a_magic_with_no_room_for_its_header_still_refuses_as_truncated() {
        let mut data = vec![0u8; 16];
        data.extend_from_slice(&AIR_WRAP_MAGIC);
        let truncated_at = data.len() - AIR_WRAP_MAGIC.len();
        assert_eq!(
            extract_air(&data).unwrap_err(),
            MtlbDecline::WrapperHeaderTruncated {
                offset: truncated_at,
                data_len: data.len(),
            }
        );
    }

    /// The container this decoder exists to refuse, built to Apple's shape.
    ///
    /// The `MTLB` header is the first 0x58 bytes of a real one with `total`
    /// (0x10) and `dir_end` (0x48) set for this length, the directory is lifted
    /// byte for byte out of `tests/fixtures/icb_object_buf.metallib` at
    /// `[0x58, 0x158)` and therefore carries that library's two function
    /// records — `mesh_main` then `object_main` — and the two wrappers follow
    /// it, `wrappers[0]` where 0x48 already points.
    fn two_function_container(wrappers: [&[u8]; 2]) -> Vec<u8> {
        /// `icb_object_buf.metallib[0x58..0x158]`.
        const DIRECTORY: [u8; 0x100] = [
            0x02, 0x00, 0x00, 0x00, 0x7f, 0x00, 0x00, 0x00, 0x4e, 0x41, 0x4d, 0x45, 0x0a, 0x00,
            0x6d, 0x65, 0x73, 0x68, 0x5f, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x54, 0x59, 0x50, 0x45,
            0x01, 0x00, 0x07, 0x48, 0x41, 0x53, 0x48, 0x20, 0x00, 0x49, 0xee, 0x71, 0x53, 0x84,
            0xff, 0x6d, 0xcd, 0x80, 0x2e, 0x69, 0x57, 0x5e, 0x7e, 0xf5, 0x29, 0xf5, 0x06, 0x22,
            0xc7, 0xac, 0x8b, 0x0a, 0x32, 0xda, 0x4d, 0xf2, 0xd8, 0xd0, 0x00, 0x33, 0x2c, 0x4d,
            0x44, 0x53, 0x5a, 0x08, 0x00, 0xc0, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4f,
            0x46, 0x46, 0x54, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x56, 0x45, 0x52, 0x53, 0x08, 0x00, 0x02, 0x00, 0x08, 0x00, 0x04, 0x00, 0x00,
            0x00, 0x45, 0x4e, 0x44, 0x54, 0x81, 0x00, 0x00, 0x00, 0x4e, 0x41, 0x4d, 0x45, 0x0c,
            0x00, 0x6f, 0x62, 0x6a, 0x65, 0x63, 0x74, 0x5f, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x54,
            0x59, 0x50, 0x45, 0x01, 0x00, 0x08, 0x48, 0x41, 0x53, 0x48, 0x20, 0x00, 0x30, 0x47,
            0xa9, 0x48, 0xd5, 0x38, 0x31, 0x85, 0xdd, 0xff, 0xc1, 0x03, 0x8a, 0x2c, 0x25, 0xc9,
            0x8d, 0x23, 0x1d, 0xd5, 0x26, 0xfb, 0x29, 0x0f, 0xa5, 0x4e, 0x33, 0x6d, 0x03, 0xe4,
            0xac, 0xe1, 0x4d, 0x44, 0x53, 0x5a, 0x08, 0x00, 0x60, 0x0e, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x4f, 0x46, 0x46, 0x54, 0x18, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x0e, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x56, 0x45, 0x52, 0x53, 0x08, 0x00, 0x02, 0x00, 0x08, 0x00,
            0x04, 0x00, 0x00, 0x00,
        ];
        // The header of a real container: only 0x10 (`total`) and 0x48
        // (`dir_end`) depend on the length, and both are written below.
        let mut header = [
            0x4d, 0x54, 0x4c, 0x42, 0x01, 0x80, 0x02, 0x00, 0x09, 0x00, 0x00, 0x81, 0x1a, 0x00,
            0x00, 0x00, 0xa0, 0x1e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x60, 0x01,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x70, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x80, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x1d, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        let dir_end = (header.len() + DIRECTORY.len()) as u64;
        let total = dir_end + wrappers[0].len() as u64 + wrappers[1].len() as u64;
        header[0x10..0x18].copy_from_slice(&total.to_le_bytes());
        header[0x48..0x50].copy_from_slice(&dir_end.to_le_bytes());
        let mut container = Vec::with_capacity(total as usize);
        container.extend_from_slice(&header);
        container.extend_from_slice(&DIRECTORY);
        container.extend_from_slice(wrappers[0]);
        container.extend_from_slice(wrappers[1]);
        container
    }

    /// A bare wrapper: four magic bytes, the 0x14-byte header, `payload`.
    fn wrapper(payload: &[u8]) -> Vec<u8> {
        let mut w = vec![0u8; WRAPPER_HEADER_LEN];
        w[0..4].copy_from_slice(&AIR_WRAP_MAGIC);
        let bitcode_size = u32::try_from(payload.len()).expect("test payload is small");
        w[8..12].copy_from_slice(&(WRAPPER_HEADER_LEN as u32).to_le_bytes());
        w[12..16].copy_from_slice(&bitcode_size.to_le_bytes());
        w.extend_from_slice(payload);
        w
    }

    /// A second function's AIR in the same container is not a second candidate
    /// to the old code — it was never a candidate at all.
    ///
    /// This is the refusal that replaces "translate whichever wrapper is first":
    /// the carve is declined by name, and the first wrapper still sits where the
    /// old scan would have taken it, so nothing about the fixture lets the old
    /// reading pass this test by accident.
    #[test]
    fn a_multi_function_container_refuses_instead_of_carving_the_first_wrapper() {
        let first = wrapper(&[1, 2, 3, 4]);
        let second = wrapper(&[5, 6, 7, 8]);
        let data = two_function_container([&first, &second]);
        assert_eq!(
            extract_air(&data).unwrap_err(),
            MtlbDecline::FunctionSelectionUnsupported {
                wrappers: 2,
                dir_end: 0x158,
                data_len: data.len(),
            }
        );
    }

    /// Two wrappers and no container at all: the count is the whole criterion,
    /// so a bare pair refuses the way a `.metallib` pair does — and reports no
    /// directory to point at.
    #[test]
    fn two_adjacent_bare_wrappers_refuse_without_a_directory() {
        let mut data = wrapper(&[1, 2, 3, 4]);
        data.extend_from_slice(&wrapper(&[5, 6, 7, 8]));
        assert_eq!(
            extract_air(&data).unwrap_err(),
            MtlbDecline::FunctionSelectionUnsupported {
                wrappers: 2,
                dir_end: 0,
                data_len: data.len(),
            }
        );
    }

    /// A directory that lists two functions while one wrapper survived.
    ///
    /// The survivor is not "the function this request meant" — the container
    /// says how many it declared — so the refusal reports the directory's count
    /// rather than the count that was found, and the carve stays refused.
    #[test]
    fn a_container_that_lost_a_wrapper_is_not_a_single_function_container() {
        let data = two_function_container([&wrapper(&[1, 2, 3, 4]), &[]]);
        assert_eq!(
            extract_air(&data).unwrap_err(),
            MtlbDecline::FunctionSelectionUnsupported {
                wrappers: 2,
                dir_end: 0x158,
                data_len: data.len(),
            }
        );
    }

    #[test]
    fn malformed_wrappers_fire_typed_declines() {
        assert_eq!(
            extract_air(&[]).unwrap_err(),
            MtlbDecline::WrappedAirMissing { data_len: 0 }
        );
        assert_eq!(
            blob_at(&[0; 8], 0).unwrap_err(),
            MtlbDecline::WrapperHeaderTruncated {
                offset: 0,
                data_len: 8
            }
        );

        let mut data = vec![0u8; WRAPPER_HEADER_LEN];
        data[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        data[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        let expected = MtlbDecline::BlobOutOfBounds {
            offset: 0,
            blob_len: u64::from(u32::MAX) * 2,
            data_len: WRAPPER_HEADER_LEN,
        };
        assert_eq!(blob_at(&data, 0).unwrap_err(), expected);
    }

    #[test]
    fn mtlb_declines_have_distinct_log_safe_reasons() {
        use crate::observe::Decline as _;
        let cases = [
            MtlbDecline::WrappedAirMissing { data_len: 1 },
            MtlbDecline::WrapperHeaderTruncated {
                offset: 1,
                data_len: 2,
            },
            MtlbDecline::BlobOutOfBounds {
                offset: 1,
                blob_len: 2,
                data_len: 3,
            },
            MtlbDecline::FunctionSelectionUnsupported {
                wrappers: 2,
                dir_end: 0x158,
                data_len: 4,
            },
        ];
        let mut slugs = std::collections::HashSet::new();
        for decline in cases {
            assert!(slugs.insert(decline.slug()));
            for (_, value) in decline.fields() {
                assert!(!value.contains(char::is_whitespace));
            }
        }
    }
}
