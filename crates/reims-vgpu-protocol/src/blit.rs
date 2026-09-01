//! Which transfer a blit opcode names.
//!
//! # Why this is a separate enumeration from the ledger
//!
//! [`crate::closure`] answers "does the device owe the guest anything for this
//! opcode, and has that been established". It says nothing about *shape*: a
//! buffer-to-buffer copy and a mipmap generation are one row each and look
//! identical from there. The model needs the shape, because the shape is what
//! decides which memory an operation touches and in which direction.
//!
//! So this is the second half of the same claim, and the two are joined by a
//! test rather than by convention: the kinds enumerated here are exactly the
//! blit-rail operations the ledger has judged and the operation vocabulary
//! classifies as transfers. An opcode that gains a contract without gaining a
//! kind fails that test, which is the only way "the vocabulary is exhaustive"
//! survives contact with a ledger that keeps changing.
//!
//! # What is deliberately absent
//!
//! The fence pair, the barrier-shaped residency and content-representation
//! records, the indirect-command-buffer family, and both `fillTexture:` forms.
//! The first three are other operation classes; the last two are unresolved and
//! must not be given a payload the model can execute, because executing a guess
//! about a write is worse than refusing it — the guest reads back content it
//! believes it wrote either way, and only the refusal says so.

use reims_vgpu_wire::ops::blit as wire;

/// The transfer an opcode names.
///
/// One variant per record shape, not per selector: the `options:` forms of
/// buffer-to-texture and texture-to-buffer share their sibling's opcode and
/// length and carry the option in room the plain form already reserves, so they
/// are the same kind. The region copy is the exception — `options:` there is a
/// different opcode at a different length — and it keeps its own variant for
/// exactly that reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BlitKind {
    /// `copyFromBuffer:…toTexture:…`, with or without `options:`.
    BufferToTexture,
    /// `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`.
    BufferToBuffer,
    /// `copyFromTexture:…toBuffer:…`, with or without `options:`.
    TextureToBuffer,
    /// `copyFromTexture:…toTexture:…` over a region.
    TextureRegion,
    /// The region copy's `options:` form, which is its own opcode and four
    /// bytes longer.
    TextureRegionOptions,
    /// `copyFromTexture:sourceSlice:sourceLevel:toTexture:…sliceCount:levelCount:`,
    /// and the whole-texture `copyFromTexture:toTexture:` that shares it.
    TextureSlices,
    /// `fillBuffer:range:value:`.
    FillBuffer,
    /// `fillBuffer:range:pattern4:`.
    FillBufferPattern4,
    /// `generateMipmapsForTexture:`.
    GenerateMipmaps,
}

impl BlitKind {
    pub const ALL: &'static [BlitKind] = &[
        BlitKind::BufferToTexture,
        BlitKind::BufferToBuffer,
        BlitKind::TextureToBuffer,
        BlitKind::TextureRegion,
        BlitKind::TextureRegionOptions,
        BlitKind::TextureSlices,
        BlitKind::FillBuffer,
        BlitKind::FillBufferPattern4,
        BlitKind::GenerateMipmaps,
    ];

    /// The opcode this kind is carried by, from the wire crate's constants.
    #[must_use]
    pub const fn wire_opcode(self) -> u32 {
        match self {
            Self::BufferToTexture => wire::OPCODE_COPY_BUFFER_TO_TEXTURE,
            Self::BufferToBuffer => wire::OPCODE_COPY_BUFFER_TO_BUFFER,
            Self::TextureToBuffer => wire::OPCODE_COPY_TEXTURE_TO_BUFFER,
            Self::TextureRegion => wire::OPCODE_COPY_TEXTURE_REGION,
            Self::TextureRegionOptions => wire::OPCODE_COPY_TEXTURE_REGION_OPTIONS,
            Self::TextureSlices => wire::OPCODE_COPY_TEXTURE_SLICES,
            Self::FillBuffer => wire::OPCODE_FILL_BUFFER,
            Self::FillBufferPattern4 => wire::OPCODE_FILL_BUFFER_PATTERN4,
            Self::GenerateMipmaps => wire::OPCODE_GENERATE_MIPMAPS,
        }
    }

    /// The kind an opcode names, or `None` if it names no transfer.
    ///
    /// `None` covers three different things — another operation class, an
    /// unresolved opcode, and an opcode with no contract at all — and this
    /// module is deliberately not the place that tells them apart. The ledger
    /// is.
    #[must_use]
    pub fn of_opcode(opcode: u32) -> Option<BlitKind> {
        BlitKind::ALL
            .iter()
            .copied()
            .find(|k| k.wire_opcode() == opcode)
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::BufferToTexture => "buffer_to_texture",
            Self::BufferToBuffer => "buffer_to_buffer",
            Self::TextureToBuffer => "texture_to_buffer",
            Self::TextureRegion => "texture_region",
            Self::TextureRegionOptions => "texture_region_options",
            Self::TextureSlices => "texture_slices",
            Self::FillBuffer => "fill_buffer",
            Self::FillBufferPattern4 => "fill_buffer_pattern4",
            Self::GenerateMipmaps => "generate_mipmaps",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::closure::{Rail, LEDGER};

    #[test]
    fn no_two_kinds_share_an_opcode() {
        for (i, a) in BlitKind::ALL.iter().enumerate() {
            for b in &BlitKind::ALL[i + 1..] {
                assert_ne!(a.wire_opcode(), b.wire_opcode(), "{a:?} and {b:?}");
            }
            assert_eq!(BlitKind::of_opcode(a.wire_opcode()), Some(*a));
        }
    }

    /// The two forms of the region copy really are two opcodes, and the two
    /// forms of the other copies really are one. Pinned because the whole
    /// variant set is shaped by that asymmetry, and a reader who assumed it was
    /// uniform would collapse the pair or split the singles.
    #[test]
    fn the_region_copy_is_the_only_split_options_form() {
        assert_ne!(
            BlitKind::TextureRegion.wire_opcode(),
            BlitKind::TextureRegionOptions.wire_opcode()
        );
        assert_eq!(
            wire::COPY_TEXTURE_REGION_OPTIONS_TOTAL_LEN - wire::COPY_TEXTURE_REGION_TOTAL_LEN,
            4
        );
        assert_eq!(
            wire::COPY_BUFFER_TO_TEXTURE_TOTAL_LEN,
            wire::COPY_TEXTURE_TO_BUFFER_TOTAL_LEN
        );
    }

    /// Every kind here is a judged blit-rail operation.
    ///
    /// The other direction — every judged transfer has a kind — cannot be
    /// asserted from this crate, because "which judged blit ops are transfers
    /// rather than fences, barriers or residency" is the operation vocabulary's
    /// classification and that lives above. `reims_vgpu_core::blit` closes it.
    #[test]
    fn every_kind_is_a_judged_blit_rail_operation() {
        for kind in BlitKind::ALL {
            let op = LEDGER
                .iter()
                .find(|o| o.rail == Rail::Blit && o.opcode == Some(kind.wire_opcode()))
                .unwrap_or_else(|| panic!("{kind:?} has no ledger row"));
            assert!(
                !op.closure.blocks_cutover(),
                "{kind:?} is {} and must not have a payload the model can execute",
                op.closure.name()
            );
        }
    }
}
