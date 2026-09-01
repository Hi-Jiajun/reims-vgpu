//! What the content-representation records ask for.
//!
//! # Eight opcodes, three questions, and one shape twice
//!
//! `optimizeContentsForCPUAccess:`, `optimizeContentsForGPUAccess:`,
//! `synchronizeResource:` and `invalidateCompressedTexture:` each come in a
//! whole-resource form and a `slice:level:` form. That is eight opcodes over
//! four directives and two granularities, and the pairing is regular enough
//! that a table is the honest way to write it — the alternative is eight arms
//! that can drift.
//!
//! # What these records are *about* is where content is, not what it is
//!
//! None of them changes a texel. They say which representation the guest is
//! about to use, or which one it no longer trusts. That makes them the
//! operations most exposed to a change of memory topology: every one of them is
//! a proven no-op today on a cell that reads "guest pages are written directly"
//! or "guest pages are the single copy of resource content", and both of those
//! are statements about the placement the current executor chose.
//!
//! The layer that gets to make placement decisions is the executor, so the
//! layer that gets to conclude "there is nothing to do" is the executor. This
//! module says what was asked.

use reims_vgpu_wire::ops::blit as wire;

/// What a content-representation record asks the device to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ContentDirective {
    /// Make the guest's own pages current for this content.
    ///
    /// The one directive with a definable semantic effect: after it, a CPU read
    /// of the guest's pages must see what the GPU produced. Whether that costs
    /// anything depends on where the content actually is.
    Synchronize,
    /// Prepare the content for CPU access.
    OptimizeForCpu,
    /// Prepare the content for GPU access.
    OptimizeForGpu,
    /// The lossless-compression metadata for this content is no longer to be
    /// trusted.
    InvalidateCompressed,
}

impl ContentDirective {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Synchronize => "synchronize",
            Self::OptimizeForCpu => "optimize_for_cpu",
            Self::OptimizeForGpu => "optimize_for_gpu",
            Self::InvalidateCompressed => "invalidate_compressed",
        }
    }
}

/// How much of a resource a record named.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ContentGranularity {
    /// The record carried a ref alone.
    WholeResource,
    /// The record carried a slice and a level.
    SliceLevel,
}

/// The directive and granularity an opcode names, if it names one.
#[must_use]
pub fn content_request(opcode: u32) -> Option<(ContentDirective, ContentGranularity)> {
    use ContentDirective::*;
    use ContentGranularity::*;
    Some(match opcode {
        wire::OPCODE_OPTIMIZE_FOR_CPU => (OptimizeForCpu, WholeResource),
        wire::OPCODE_OPTIMIZE_FOR_GPU => (OptimizeForGpu, WholeResource),
        wire::OPCODE_SYNCHRONIZE_RESOURCE => (Synchronize, WholeResource),
        wire::OPCODE_INVALIDATE_COMPRESSED_TEXTURE => (InvalidateCompressed, WholeResource),
        wire::OPCODE_OPTIMIZE_FOR_CPU_SLICE_LEVEL => (OptimizeForCpu, SliceLevel),
        wire::OPCODE_OPTIMIZE_FOR_GPU_SLICE_LEVEL => (OptimizeForGpu, SliceLevel),
        wire::OPCODE_SYNCHRONIZE_TEXTURE => (Synchronize, SliceLevel),
        wire::OPCODE_INVALIDATE_COMPRESSED_TEXTURE_SLICE_LEVEL => {
            (InvalidateCompressed, SliceLevel)
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::closure::{Closure, Rail, LEDGER};

    /// The two granularities cover the same four directives, and the wire
    /// crate's own predicates agree about which opcodes are which shape.
    #[test]
    fn each_directive_has_both_granularities_and_the_shapes_agree() {
        for directive in [
            ContentDirective::Synchronize,
            ContentDirective::OptimizeForCpu,
            ContentDirective::OptimizeForGpu,
            ContentDirective::InvalidateCompressed,
        ] {
            let mut seen = [false; 2];
            for opcode in 0u32..0x200 {
                let Some((d, g)) = content_request(opcode) else {
                    continue;
                };
                if d != directive {
                    continue;
                }
                match g {
                    ContentGranularity::WholeResource => {
                        assert!(wire::is_ref(opcode), "{opcode:#x} is not a bare ref record");
                        seen[0] = true;
                    }
                    ContentGranularity::SliceLevel => {
                        assert!(
                            wire::is_ref_slice_level(opcode),
                            "{opcode:#x} is not a slice/level record"
                        );
                        seen[1] = true;
                    }
                }
            }
            assert_eq!(seen, [true, true], "{directive:?} is missing a form");
        }
    }

    /// Every opcode named here is a judged blit-rail operation, and every one
    /// is a proven no-op.
    ///
    /// The second half is the part worth asserting: it is the premise of this
    /// module's documentation, and of `reims_vgpu_core::resource_state` keeping
    /// these as operations rather than dropping them. If one of them ever
    /// becomes implemented or unresolved, that reasoning needs reading again.
    #[test]
    fn every_content_request_is_a_judged_proven_no_op() {
        let mut count = 0;
        for opcode in 0u32..0x200 {
            if content_request(opcode).is_none() {
                continue;
            }
            count += 1;
            let op = LEDGER
                .iter()
                .find(|o| o.rail == Rail::Blit && o.opcode == Some(opcode))
                .unwrap_or_else(|| panic!("{opcode:#x} has no ledger row"));
            assert!(
                matches!(op.closure, Closure::ProvenNoOp { .. }),
                "{opcode:#x} is {}",
                op.closure.name()
            );
        }
        assert_eq!(count, 8, "four directives in two granularities");
    }
}
