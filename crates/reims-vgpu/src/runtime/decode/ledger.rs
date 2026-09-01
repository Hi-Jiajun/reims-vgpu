//! Does the closure ledger describe *these* decoders?
//!
//! `reims_vgpu_protocol::closure` records one outcome per decodable operation,
//! and its own tests keep the row set equal to the serializer's selector
//! manifest. Neither of those touches this crate. So the ledger could say a
//! render opcode is implemented while this rail's decoder refuses it as
//! unknown, and nothing would notice: one is a statement about the protocol,
//! the other is a match arm, and until now they were only ever read by people.
//!
//! These tests are the join. They drive the real per-rail decoders across every
//! ledger row and assert the two agree about which operations exist. They say
//! nothing about whether an outcome is the *right* one — no test can — only
//! that the ledger and the decoders are describing the same opcode space.
//!
//! # What "recognised" means here
//!
//! Each rail refuses an opcode outside its own accepted window with a distinct
//! status (`ErrUnknownOpcode`, `ErrUnsupportedOpcode`), separately from
//! refusing a record whose payload is too short for its layout (`ErrShort`,
//! `ErrBadLength`). Only the first pair is a claim about the opcode, so that is
//! what is asserted: a generously sized zero payload is offered and the decoder
//! may still refuse its *contents*, because a record of zeroes is not a record
//! any of these selectors would actually write.

use reims_vgpu_protocol::closure::{Rail, LEDGER};

/// A record header for `opcode` followed by `payload` zero bytes.
fn zero_record(opcode: u32, payload: usize) -> Vec<u8> {
    let total = reims_vgpu_wire::OP_HEADER_LEN + payload;
    let mut v = vec![0u8; total];
    v[0..4].copy_from_slice(&opcode.to_le_bytes());
    v[4..8].copy_from_slice(&(total as u32).to_le_bytes());
    v
}

/// Wide enough for every head in these families plus a counted entry array;
/// the point is to reach the opcode arm, not to satisfy a layout.
const GENEROUS_PAYLOAD: usize = 256;

#[test]
fn the_render_decoder_recognises_every_render_operation_the_ledger_records() {
    use super::render::{decode, DecodeStatus};
    for op in LEDGER
        .iter()
        .filter(|o| o.rail == Rail::Render)
        .filter_map(|o| o.opcode)
    {
        let refused_the_opcode = matches!(
            decode(&zero_record(op, GENEROUS_PAYLOAD)),
            Err(DecodeStatus::ErrUnknownOpcode) | Err(DecodeStatus::ErrUnsupportedOpcode)
        );
        assert!(
            !refused_the_opcode,
            "the closure ledger records render {op:#x} and this rail's decoder \
             refuses the opcode itself: one of the two is describing an \
             operation the other says does not exist"
        );
    }
}

#[test]
fn the_compute_decoder_recognises_every_compute_operation_the_ledger_records() {
    use super::compute::{decode, DecodeStatus};
    for op in LEDGER
        .iter()
        .filter(|o| o.rail == Rail::Compute)
        .filter_map(|o| o.opcode)
    {
        let refused_the_opcode = matches!(
            decode(&zero_record(op, GENEROUS_PAYLOAD)),
            Err(DecodeStatus::ErrUnknownOpcode) | Err(DecodeStatus::ErrUnsupportedOpcode)
        );
        assert!(
            !refused_the_opcode,
            "the closure ledger records compute {op:#x} and this rail's decoder \
             refuses the opcode itself"
        );
    }
}

#[test]
fn the_blit_decoder_recognises_every_blit_operation_the_ledger_records() {
    use super::blit::{decode, DecodeStatus};
    for op in LEDGER
        .iter()
        .filter(|o| o.rail == Rail::Blit)
        .filter_map(|o| o.opcode)
    {
        let refused_the_opcode = matches!(
            decode(&zero_record(op, GENEROUS_PAYLOAD)),
            Err(DecodeStatus::ErrUnknownOpcode)
        );
        assert!(
            !refused_the_opcode,
            "the closure ledger records blit {op:#x} and this rail's decoder \
             refuses the opcode itself"
        );
    }
}

/// The other direction on the one rail that can state its own window.
///
/// The render decoder accepts a contiguous opcode range and falls through to
/// `Kind::OtherAccepted` inside it, so an opcode can be *accepted* without any
/// arm claiming it — which is a decodable operation the device drops. Those are
/// numbers rather than known selectors, so the ledger does not carry rows for
/// them; what it must not do is disagree about the window's edge, because an
/// operation the ledger judges must be inside it.
#[test]
fn no_ledger_operation_sits_outside_the_render_encoder_window() {
    use super::render::opcode_above_the_encoder_window;
    for op in LEDGER
        .iter()
        .filter(|o| o.rail == Rail::Render)
        .filter_map(|o| o.opcode)
    {
        assert!(
            !opcode_above_the_encoder_window(op),
            "render {op:#x} is judged by the ledger and above the window this \
             rail accepts, so the judgement can never be acted on"
        );
    }
}
