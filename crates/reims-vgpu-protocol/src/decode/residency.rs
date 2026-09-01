//! Lifting the residency declarations.
//!
//! # Four records, two subjects, and no shared head
//!
//! `useResource:usage:stages:` is `0x89`, with `usage` and `stages` sharing the
//! word at `+4` at sixteen bits each. `useHeap:stages:` is `0x1b`, has no usage
//! at all, and puts `stages` alone at `+4` — so its refs begin at `+6`, an
//! odd-of-four offset. The unqualified forms are `0x86` and `0x87`, inherited
//! from the encoder base class rather than declared on any one encoder, and
//! they carry no stages: `0x86` is a bare count, `0x87` is a count and a usage
//! widened back to thirty-two bits.
//!
//! So the same declaration arrives in four head shapes and the usage field is
//! sixteen bits in one and thirty-two in another. Both reach [`ResourceUsage`]
//! here, because the semantic question does not change with the width — and the
//! *absence* of stages stays `None`, because a selector without the argument
//! and a guest passing zero are different facts.
//!
//! # A heap is not a resource
//!
//! Two of the four name heaps and two name resources, and the record set is the
//! only place that says which. A heap declaration makes every resource
//! allocated from that heap resident; a resource declaration names the
//! resources themselves. Collapsing the two would either declare far too much
//! or declare a heap's ref as though it were a texture's.

use super::{no_record, short, DecodeRefusal};
use crate::closure::Rail;
use crate::residency::{RenderStages, ResourceUsage};
use reims_vgpu_wire::op::Op;
use reims_vgpu_wire::ops::render as wire;
use reims_vgpu_wire::ops::render::RefBind;

/// What a residency record declares resident.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResidencySubject {
    /// The refs are heaps; everything allocated from them becomes resident.
    Heaps,
    /// The refs are the resources themselves.
    Resources,
}

/// One lifted residency declaration.
///
/// The refs stay a window into the guest's bytes: the list is guest-sized and
/// the model appends resolved ids into the transaction's own resource arena,
/// so a copy here would be made before the ids exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyRecord<'a> {
    pub subject: ResidencySubject,
    pub refs: &'a [RefBind],
    /// The usage the record declared, or `None` when its selector has none.
    ///
    /// A heap declaration carries no usage on either of its two forms.
    pub usage: Option<ResourceUsage>,
    /// The stages the record declared, or `None` when its selector has none.
    pub stages: Option<RenderStages>,
}

/// Whether an opcode is a residency declaration on `rail`.
///
/// The unqualified pair is inherited by every encoder, so it is not the render
/// rail's alone — asking with the rail is what keeps that from being read as
/// "the render encoder declares them and the compute one does not".
#[must_use]
pub fn is_residency(rail: Rail, opcode: u32) -> bool {
    match rail {
        Rail::Render => matches!(
            opcode,
            wire::OPCODE_USE_HEAP
                | wire::OPCODE_USE_RESOURCE
                | wire::OPCODE_USE_HEAPS_NO_STAGES
                | wire::OPCODE_USE_RESOURCES_NO_STAGES
        ),
        Rail::Compute => matches!(
            opcode,
            wire::OPCODE_USE_HEAPS_NO_STAGES | wire::OPCODE_USE_RESOURCES_NO_STAGES
        ),
        _ => false,
    }
}

/// Lift a residency declaration out of its bytes.
pub fn decode<'a>(rail: Rail, op: &Op<'a>) -> Result<ResidencyRecord<'a>, DecodeRefusal> {
    let opcode = op.opcode();
    if !is_residency(rail, opcode) {
        return Err(no_record(rail, opcode));
    }
    Ok(match opcode {
        wire::OPCODE_USE_RESOURCE => {
            let (head, refs) = wire::use_resource(op)
                .map_err(|_| counted(rail, op, core::mem::size_of::<wire::UseResource>()))?;
            ResidencyRecord {
                subject: ResidencySubject::Resources,
                refs,
                usage: Some(ResourceUsage(u32::from(head.usage.get()))),
                stages: Some(RenderStages(u32::from(head.stages.get()))),
            }
        }
        wire::OPCODE_USE_HEAP => {
            let (head, refs) = wire::use_heap(op)
                .map_err(|_| counted(rail, op, core::mem::size_of::<wire::UseHeap>()))?;
            ResidencyRecord {
                subject: ResidencySubject::Heaps,
                refs,
                usage: None,
                stages: Some(RenderStages(u32::from(head.stages.get()))),
            }
        }
        wire::OPCODE_USE_HEAPS_NO_STAGES => {
            let (_, refs) = wire::use_heaps_no_stages(op)
                .map_err(|_| counted(rail, op, core::mem::size_of::<wire::UseHeapsNoStages>()))?;
            ResidencyRecord {
                subject: ResidencySubject::Heaps,
                refs,
                usage: None,
                stages: None,
            }
        }
        _ => {
            let (head, refs) = wire::use_resources_no_stages(op).map_err(|_| {
                counted(rail, op, core::mem::size_of::<wire::UseResourcesNoStages>())
            })?;
            ResidencyRecord {
                subject: ResidencySubject::Resources,
                refs,
                usage: Some(ResourceUsage(head.usage.get())),
                stages: None,
            }
        }
    })
}

/// A counted record that did not fit. The count leads every head here, which is
/// the one thing the four shapes do agree on.
fn counted(rail: Rail, op: &Op<'_>, head_len: usize) -> DecodeRefusal {
    let have = op.payload.len();
    if have < head_len {
        return short(rail, op.opcode(), have, head_len);
    }
    let mut count = [0u8; 4];
    count.copy_from_slice(&op.payload[..4]);
    DecodeRefusal::CountOverruns {
        rail,
        opcode: op.opcode(),
        count: u32::from_le_bytes(count),
        have,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use reims_vgpu_wire::op::{op, OP_HEADER_LEN};

    fn record(opcode: u32, payload: &[u8]) -> Vec<u8> {
        let total = (OP_HEADER_LEN + payload.len()) as u32;
        let mut out = Vec::with_capacity(total as usize);
        out.extend_from_slice(&opcode.to_le_bytes());
        out.extend_from_slice(&total.to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn lift(rail: Rail, bytes: &[u8]) -> Result<ResidencyRecord<'_>, DecodeRefusal> {
        decode(rail, &op(bytes, 0).expect("framed"))
    }

    /// The heap record's refs begin at `+6`, not `+8`. The serializer sizes the
    /// record as though it had a usage field and then does not write one, so a
    /// head derived from the length starts every ref two bytes late and reads
    /// each one straddling two entries. Two distinct refs are what makes that
    /// visible.
    #[test]
    fn the_heap_records_refs_begin_where_its_head_ends_and_not_where_its_length_suggests() {
        let mut payload = 2u32.to_le_bytes().to_vec();
        payload.extend_from_slice(&2u16.to_le_bytes());
        payload.extend_from_slice(&6565u32.to_le_bytes());
        payload.extend_from_slice(&6666u32.to_le_bytes());
        // The two bytes the serializer sizes for and never writes.
        payload.extend_from_slice(&[0xaa, 0xaa]);
        let bytes = record(wire::OPCODE_USE_HEAP, &payload);
        let lifted = lift(Rail::Render, &bytes).expect("lifted");
        assert_eq!(lifted.subject, ResidencySubject::Heaps);
        assert_eq!(lifted.refs.len(), 2);
        assert_eq!(lifted.refs[0].object_ref.get(), 6565);
        assert_eq!(lifted.refs[1].object_ref.get(), 6666);
        assert_eq!(lifted.usage, None);
        assert_eq!(lifted.stages, Some(RenderStages(2)));
    }

    /// The usage field is sixteen bits in the `stages:` form and thirty-two in
    /// the unqualified one, and both reach the same type. The stages stay
    /// `None` on the form whose selector has none.
    #[test]
    fn the_two_usage_widths_reach_one_type_and_only_one_form_has_stages() {
        let mut qualified = 1u32.to_le_bytes().to_vec();
        qualified.extend_from_slice(&ResourceUsage::WRITE.to_le_bytes()[..2]);
        qualified.extend_from_slice(&2u16.to_le_bytes());
        qualified.extend_from_slice(&5151u32.to_le_bytes());
        let bytes = record(wire::OPCODE_USE_RESOURCE, &qualified);
        let lifted = lift(Rail::Render, &bytes).expect("lifted");
        assert_eq!(lifted.usage, Some(ResourceUsage(ResourceUsage::WRITE)));
        assert_eq!(lifted.stages, Some(RenderStages(2)));

        let mut unqualified = 1u32.to_le_bytes().to_vec();
        unqualified.extend_from_slice(&ResourceUsage::WRITE.to_le_bytes());
        unqualified.extend_from_slice(&5151u32.to_le_bytes());
        let bytes = record(wire::OPCODE_USE_RESOURCES_NO_STAGES, &unqualified);
        let lifted = lift(Rail::Render, &bytes).expect("lifted");
        assert_eq!(lifted.usage, Some(ResourceUsage(ResourceUsage::WRITE)));
        assert_eq!(lifted.stages, None);
    }

    /// A heap declaration and a resource declaration stay apart. They name
    /// different objects, and reading one as the other either makes a heap's
    /// ref into a texture's or declares a heap's whole contents where the guest
    /// named one buffer.
    #[test]
    fn heaps_and_resources_are_different_subjects() {
        let mut payload = 1u32.to_le_bytes().to_vec();
        payload.extend_from_slice(&6565u32.to_le_bytes());
        let bytes = record(wire::OPCODE_USE_HEAPS_NO_STAGES, &payload);
        let heaps = lift(Rail::Render, &bytes).expect("lifted");
        assert_eq!(heaps.subject, ResidencySubject::Heaps);

        let mut resources = 1u32.to_le_bytes().to_vec();
        resources.extend_from_slice(&0u32.to_le_bytes());
        resources.extend_from_slice(&6565u32.to_le_bytes());
        let bytes = record(wire::OPCODE_USE_RESOURCES_NO_STAGES, &resources);
        let lifted = lift(Rail::Render, &bytes).expect("lifted");
        assert_eq!(lifted.subject, ResidencySubject::Resources);
    }

    /// The unqualified pair is inherited by every encoder, so the compute rail
    /// carries it too — while the `stages:`-qualified pair is the render
    /// encoder's own and means nothing on compute.
    #[test]
    fn the_inherited_pair_reaches_both_encoders_and_the_declared_pair_does_not() {
        let mut payload = 1u32.to_le_bytes().to_vec();
        payload.extend_from_slice(&6565u32.to_le_bytes());
        assert!(lift(
            Rail::Compute,
            &record(wire::OPCODE_USE_HEAPS_NO_STAGES, &payload)
        )
        .is_ok());
        assert!(lift(Rail::Compute, &record(wire::OPCODE_USE_HEAP, &payload)).is_err());
        assert!(lift(Rail::Blit, &record(wire::OPCODE_USE_HEAP, &payload)).is_err());
    }

    /// A count larger than the record it sits in is reported with both numbers.
    #[test]
    fn a_declaration_longer_than_its_record_reports_the_count_and_the_bytes() {
        let mut payload = 200u32.to_le_bytes().to_vec();
        payload.extend_from_slice(&5151u32.to_le_bytes());
        assert_eq!(
            lift(
                Rail::Render,
                &record(wire::OPCODE_USE_HEAPS_NO_STAGES, &payload)
            ),
            Err(DecodeRefusal::CountOverruns {
                rail: Rail::Render,
                opcode: wire::OPCODE_USE_HEAPS_NO_STAGES,
                count: 200,
                have: payload.len(),
            })
        );
    }

    /// Every residency opcode in the ledger lifts a record, on every rail the
    /// ledger gives it a row on.
    #[test]
    fn every_ledger_residency_row_lifts_a_record() {
        let mut payload = 1u32.to_le_bytes().to_vec();
        payload.extend_from_slice(&[0u8; 16]);
        let mut seen = 0usize;
        for rail in [Rail::Render, Rail::Compute, Rail::Blit] {
            for opcode in 0u32..0x200 {
                if !is_residency(rail, opcode) {
                    continue;
                }
                seen += 1;
                assert!(
                    lift(rail, &record(opcode, &payload)).is_ok(),
                    "{rail:?} {opcode:#x}"
                );
            }
        }
        assert_eq!(seen, 6);
    }
}
