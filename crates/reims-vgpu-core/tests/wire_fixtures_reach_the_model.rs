//! Run Apple's own records through the model's decode-and-resolve path.
//!
//! `reims-vgpu-wire` captures what the serializer emits and pins its layouts;
//! `reims-vgpu-protocol` lifts those layouts into records and this crate turns
//! them into operations. This is the only test that puts the whole path against
//! bytes Apple actually produced.
//!
//! # The claim
//!
//! **A record Apple's serializer produced is never declined for a reason about
//! its shape.** The model may decline an operation whose ledger row is open, or
//! settled as a refusal, or absent — those are contract answers, they are
//! reported below, and they are not failures. What it may not do is call a
//! well-formed record short, over-long or over-counted, because that is the
//! guest's work lost to a layout this project has wrong.
//!
//! The distinction is the whole design. `reims-vgpu`'s equivalent test found
//! four real divergences that way, and each of them looked exactly like the
//! contract answers on either side of it until the two were separated.
//!
//! # The second claim, which a single decode cannot make
//!
//! A capture arena is zero-filled exactly where a guest's command ring is not,
//! so a field read *wider* than the serializer writes agrees with the fixture
//! by accident. The oracle measures a per-bit `written_mask` — every case
//! captured twice under complementary fills — and
//! [`the_model_reads_no_bit_apples_serializer_never_wrote`] repaints every
//! unwritten bit to all-zero and then to all-one and requires the same resolved
//! operation both times. A model whose answer moves read a byte the serializer
//! never wrote, which in a guest is stale ring rather than data.
//!
//! Both tests share one call into the model, and here there is no second copy
//! to write: `resolve::operation` has a single entry and the ledger inside it
//! picks the decoder.
//!
//! Fixtures are not committed. With none present both tests are `ignored`,
//! decided at build time by `build.rs`.

#![cfg(wire_fixtures)]

use reims_vgpu_core::exec::{ExecArenas, ResolvedOperation};
use reims_vgpu_core::identity::{ObjectListRef, ResourceId, SlotGeneration};
use reims_vgpu_core::resolve::{operation, RefResolver, ResolveRefusal};
use reims_vgpu_protocol::closure::{Rail, LEDGER};
use reims_vgpu_protocol::decode::{op, DecodeRefusal};
use reims_vgpu_testkit::{fixtures, unhex};

/// A resolver that answers every ref.
///
/// Resolution is not what this test is about: a fixture's refs are the oracle
/// stub's, no object table exists here, and an unresolvable ref would report a
/// missing object where the question is about bytes. Answering everything means
/// every refusal that does reach the assertions came from the record.
struct Everything;

impl RefResolver for Everything {
    fn resource(&self, object_ref: u32) -> Option<ResourceId> {
        Some(ResourceId {
            slot: ObjectListRef(object_ref),
            generation: SlotGeneration(1),
        })
    }
}

/// What the model made of one record.
enum Verdict {
    /// It became an operation.
    Resolved(ResolvedOperation),
    /// The ledger declined it: no row, an open row, or a settled refusal. A
    /// contract answer, reported and not failed.
    Declined(&'static str),
    /// The model refused a record the serializer produced, for a reason about
    /// the record's shape. That is a layout this project has wrong.
    WrongShape(&'static str),
}

/// Read one record the way production would.
fn read(rail: Rail, bytes: &[u8]) -> Verdict {
    let Ok(view) = op(bytes, 0) else {
        // The framing itself did not parse, which is a shape claim about a
        // buffer the serializer wrote.
        return Verdict::WrongShape("op_framing_refused");
    };
    let mut arenas = ExecArenas::default();
    match operation(rail, &view, &Everything, &mut arenas) {
        Ok(resolved) => Verdict::Resolved(resolved),
        Err(ResolveRefusal::Decode(
            refusal @ (DecodeRefusal::UnknownOpcode { .. }
            | DecodeRefusal::Unjudged { .. }
            | DecodeRefusal::RefusedByContract { .. }),
        )) => Verdict::Declined(refusal.reason()),
        Err(ResolveRefusal::Decode(refusal)) => Verdict::WrongShape(refusal.reason()),
        Err(refusal) => Verdict::WrongShape(refusal.reason()),
    }
}

/// Whether a case's selector wrote a segment header rather than a record.
///
/// A segment header has **no opcode**: it is the eight bytes
/// `-beginSegment:protectionOptions:` writes, and it reaches the model through
/// `ExecBuilder::begin_segment` rather than through `resolve::operation`.
/// Routing it by selector is what keeps a boundary from being read as whatever
/// record its first four bytes happen to spell — which for a segment header is
/// its own length, and for a zero-length one is opcode zero.
fn is_boundary(selector: &str) -> bool {
    selector.starts_with("beginSegment")
}

/// One captured record: what it was called, what emitted it, and its bytes
/// beside the per-bit mask of which of them the serializer actually wrote.
struct Case {
    name: String,
    selector: String,
    rail: Rail,
    bytes: Vec<u8>,
    written_mask: Vec<u8>,
}

/// Every fixture that is a record rather than a boundary.
///
/// Cases whose class names no rail are counted rather than silently dropped:
/// `PGSerializerCommandEncoder` is the shared base class, and a record captured
/// on it was emitted by whichever encoder inherited the selector — the capture
/// does not say which, so there is no rail to dispatch on.
fn cases() -> Vec<Case> {
    let json = fixtures();
    let mut skipped_base_class = 0usize;
    let mut out = Vec::new();
    for case in json["cases"].as_array().expect("cases is an array") {
        let name = case["name"].as_str().expect("name").to_string();
        let class = case["class"].as_str().expect("class");
        let selector = case["selector"].as_str().expect("selector").to_string();
        if is_boundary(&selector) {
            continue;
        }
        let Some(rail) = Rail::from_class(class) else {
            skipped_base_class += 1;
            continue;
        };
        out.push(Case {
            name,
            selector,
            rail,
            bytes: unhex(case["buffer"].as_str().expect("buffer")),
            written_mask: unhex(case["written_mask"].as_str().expect("written_mask")),
        });
    }
    assert!(
        !out.is_empty(),
        "fixtures were present and carried no cases"
    );
    println!(
        "cases: {} records, {skipped_base_class} on the shared base class",
        out.len()
    );
    out
}

/// Every boundary record Apple wrote parses, and names a segment the model has.
///
/// Boundaries are checked here rather than dropped from the sweep. One carries
/// no opcode and cannot go through `resolve::operation`, but "no opcode" is
/// exactly why it must not be silently skipped: a header read as a record would
/// be read as whatever its length field spells, and for a segment that has not
/// ended yet that is opcode zero.
///
/// **A protected begin-segment writes three records, and only two of them are
/// headers.** The type-5 envelope header, then its eight-byte payload carrying
/// the options word, then the encoder's own header. The payload is eight bytes
/// like a header, and read as one it would call itself a render segment — the
/// options word `0x44` lands where a header's length goes and zero lands where
/// its type does.
///
/// What tells them apart is the measured written mask, not a field: a header
/// allocates eight bytes and writes **seven**, and the envelope's payload
/// writes all eight. That is the same measurement the wire crate's own poison
/// sweep produces, and it is a property of what the serializer did rather than
/// of how a fixture was named.
#[test]
fn every_boundary_record_apple_wrote_names_a_segment_the_model_has() {
    use reims_vgpu_protocol::segment::{segment_role, SegmentRole};
    let json = fixtures();
    let mut headers = 0usize;
    let mut envelopes = 0usize;
    for case in json["cases"].as_array().expect("cases is an array") {
        let selector = case["selector"].as_str().expect("selector");
        if !is_boundary(selector) {
            continue;
        }
        let name = case["name"].as_str().expect("name");
        let class = case["class"].as_str().expect("class");
        let rail = Rail::from_class(class).unwrap_or_else(|| panic!("{name}: {class} has no rail"));
        let bytes = unhex(case["buffer"].as_str().expect("buffer"));
        let mask = unhex(case["written_mask"].as_str().expect("written_mask"));
        let expect = &case["expect"];

        if mask.iter().all(|&byte| byte == 0xff) {
            // The envelope's payload: one word, and every byte of it written.
            let envelope = reims_vgpu_wire::ops::segment::protection_options_envelope(&bytes)
                .unwrap_or_else(|e| panic!("{name}: the envelope payload did not parse: {e:?}"));
            assert_eq!(
                envelope.protection_options.get(),
                expect["protection_options"].as_u64().expect("options"),
                "{name}: the envelope carried a different options word"
            );
            envelopes += 1;
            continue;
        }

        let header = reims_vgpu_wire::ops::segment::segment_header(&bytes)
            .unwrap_or_else(|e| panic!("{name}: a segment header did not parse: {e:?}"));
        if let Some(expected_type) = expect["segment_type"].as_u64() {
            assert_eq!(
                u64::from(header.segment_type),
                expected_type,
                "{name}: the header's type is not the one the capture asked for"
            );
        }
        let role = segment_role(header.segment_type).unwrap_or_else(|| {
            panic!("{name}: segment type {} names nothing", header.segment_type)
        });
        // A capture on an encoder class must produce that encoder's segment,
        // not merely *a* segment: reading the type off the wrong byte would
        // still land on a valid role. The envelope header is the exception —
        // it is the protection envelope and belongs to no encoder.
        if let SegmentRole::Encoder(kind) = role {
            assert_eq!(
                kind.rail(),
                rail,
                "{name}: segment kind disagrees with the capture class"
            );
        }
        headers += 1;
    }
    println!("boundaries: {headers} headers, {envelopes} envelope payloads");
    assert!(headers > 0, "no segment header reached this test");
    assert!(envelopes > 0, "no protection envelope reached this test");
}

/// The root rail is object lifecycle, not a command stream: creations and
/// deletions reach the model through a different transaction with no encoder,
/// so a stream decoder declining them is correct rather than a gap.
fn is_stream_rail(rail: Rail) -> bool {
    !matches!(rail, Rail::Root)
}

/// No record Apple produced is declined for its shape.
#[test]
fn no_record_apples_serializer_produced_is_refused_for_its_shape() {
    let mut resolved = 0usize;
    let mut declined: std::collections::BTreeMap<&str, Vec<String>> =
        std::collections::BTreeMap::new();
    let mut wrong = Vec::new();

    for case in cases() {
        let Case {
            name,
            selector,
            rail,
            bytes,
            ..
        } = case;
        if !is_stream_rail(rail) {
            continue;
        }
        match read(rail, &bytes) {
            Verdict::Resolved(_) => resolved += 1,
            Verdict::Declined(reason) => {
                if reason == "decode_opcode_unknown" {
                    // The ledger is keyed by opcode, and six of its rows carry
                    // none: the blit `withCommand:` selectors write their
                    // command argument *into* the opcode field, and the info
                    // rail's coordinate mapping is a request rather than a
                    // record. A lookup by opcode cannot reach those rows, so
                    // "no row" here must mean "an opcode-less row", and this is
                    // what says so. Anything else is an opcode Apple emits that
                    // the ledger has never heard of.
                    assert!(
                        LEDGER.iter().any(|row| row.rail == rail
                            && row.opcode.is_none()
                            && row.selector.split("; ").any(|s| s == selector)),
                        "{name}: {selector} has no ledger row at all"
                    );
                }
                declined.entry(reason).or_default().push(name.clone());
            }
            Verdict::WrongShape(reason) => wrong.push(format!("{name} ({rail:?}): {reason}")),
        }
    }

    println!("resolved: {resolved}");
    for (reason, names) in &declined {
        println!("declined: {reason} x{}", names.len());
    }
    if let Some(names) = declined.get("decode_opcode_unknown") {
        println!("reached only by selector: {}", names.join(", "));
    }
    assert!(
        wrong.is_empty(),
        "records the serializer produced were refused for their shape:\n  {}",
        wrong.join("\n  ")
    );
    assert!(resolved > 0, "no fixture reached the model at all");
}

/// The model's answer does not move when bytes the serializer never wrote do.
#[test]
fn the_model_reads_no_bit_apples_serializer_never_wrote() {
    let mut checked = 0usize;
    let mut moved = Vec::new();

    for case in cases() {
        let Case {
            name,
            rail,
            bytes,
            written_mask: mask,
            ..
        } = case;
        if !is_stream_rail(rail) {
            continue;
        }
        assert_eq!(
            mask.len(),
            bytes.len(),
            "{name}: the written mask and the buffer are different lengths"
        );
        // Every bit the mask says was not written, painted both ways. The two
        // fills are complements, so nothing can agree by accident.
        let mut zeroed = bytes.clone();
        let mut oned = bytes.clone();
        for (index, &written) in mask.iter().enumerate() {
            zeroed[index] &= written;
            oned[index] |= !written;
        }
        if zeroed == oned {
            // Every bit was written; there is nothing to repaint.
            continue;
        }
        checked += 1;
        let a = signature(read(rail, &zeroed));
        let b = signature(read(rail, &oned));
        if a != b {
            moved.push(format!("{name} ({rail:?}): {a} vs {b}"));
        }
    }

    println!("records with unwritten bits: {checked}");
    assert!(
        moved.is_empty(),
        "the model read a bit the serializer never wrote:\n  {}",
        moved.join("\n  ")
    );
}

/// A verdict rendered for comparison against the same record's other fill.
///
/// Nothing parses this. It only has to be faithful and stable, and the resolved
/// operation's own `Debug` is both — it prints every field the model kept, so a
/// field that moved with an unwritten bit shows up as a difference here.
fn signature(verdict: Verdict) -> String {
    match verdict {
        Verdict::Resolved(op) => format!("resolved {op:?}"),
        Verdict::Declined(reason) => format!("declined {reason}"),
        Verdict::WrongShape(reason) => format!("wrong-shape {reason}"),
    }
}

/// Every record Apple produced can be placed in the encoder that emitted it.
///
/// The tests above stop at [`operation`]: they prove a record decodes and
/// resolves. This one carries it one link further and puts it through
/// [`ExecBuilder`], which is the only way a resolved operation ever reaches a
/// transaction — `begin_segment`, `record`, `end_segment`, `finish`.
///
/// That link asks a question resolution cannot. [`ExecBuilder::record`] routes
/// on the operation's own [`OperationClass`]: five classes name exactly one
/// rail and are refused inside any other segment, and the rest are admitted by
/// whichever encoder is open only if that encoder carries the class at all. So
/// an operation whose class disagrees with the encoder class the capture came
/// from is `RailMismatch` here — and it is *nothing* in the tests above,
/// because `resolve::operation` is handed the rail rather than deriving it.
///
/// The rail is the fixture's, from the serializer class that emitted the
/// record. That is the whole point: the model is being asked to admit each
/// record where Apple actually wrote it, not where the model would have
/// preferred it.
///
/// A record the ledger declines is skipped rather than failed, for the same
/// reason it is above — a contract answer is not a defect. What must not
/// happen is a resolved operation the model then refuses to place.
#[test]
fn every_record_apple_produced_is_one_the_model_can_place_where_it_was_written() {
    use reims_vgpu_core::exec::ExecBuilder;
    use reims_vgpu_core::identity::{
        ChannelId, ChannelSequence, IngressOrdinal, SessionGeneration,
    };
    use reims_vgpu_protocol::segment::SegmentKind;

    let mut placed = 0usize;
    let mut misplaced = Vec::new();
    let mut per_rail: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();

    for case in cases() {
        let Case {
            name, rail, bytes, ..
        } = case;
        if !is_stream_rail(rail) {
            continue;
        }
        let Verdict::Resolved(op) = read(rail, &bytes) else {
            continue;
        };
        // Every stream rail has a segment kind — `of_rail` answers `None` only
        // for the root, which `is_stream_rail` already excluded.
        let kind = SegmentKind::of_rail(rail).expect("a stream rail names a segment");

        let mut builder = ExecBuilder::new(
            SessionGeneration::FIRST,
            ChannelId(0),
            ChannelSequence(placed as u64),
            IngressOrdinal(placed as u64),
        );
        let mut place = || -> Result<(), String> {
            builder
                .begin_segment(kind.wire_type(), false)
                .map_err(|r| format!("begin: {}", r.reason()))?;
            builder
                .record(op)
                .map_err(|r| format!("record: {}", r.reason()))?;
            builder
                .end_segment()
                .map_err(|r| format!("end: {}", r.reason()))?;
            Ok(())
        };
        match place() {
            Ok(()) => {
                let tx = builder
                    .finish()
                    .expect("an encoder that began and ended leaves nothing open");
                assert_eq!(
                    tx.record_count(),
                    1,
                    "{name}: the record was accepted and then not carried"
                );
                placed += 1;
                *per_rail.entry(format!("{rail:?}")).or_default() += 1;
            }
            Err(why) => misplaced.push(format!("{name} ({rail:?}, {kind:?}): {why}")),
        }
    }

    for (rail, count) in &per_rail {
        println!("placed: {rail} x{count}");
    }
    assert!(
        misplaced.is_empty(),
        "records the serializer produced could not be placed in the encoder that wrote them:\n  {}",
        misplaced.join("\n  ")
    );
    assert!(placed > 0, "no fixture reached a transaction at all");
}
