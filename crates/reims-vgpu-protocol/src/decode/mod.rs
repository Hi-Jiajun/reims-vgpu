//! Records lifted out of bytes, with the guest's own names still on them.
//!
//! # Two steps, and this is the first
//!
//! Turning a packet into work is two questions and they have different owners.
//! *What did the guest write* is a wire question with a contract answer, and it
//! is this. *Which object does this ref mean, and what is its content version*
//! is a question about device state, and it belongs to `reims-vgpu-core`, which
//! has the registries.
//!
//! Splitting them is what stops the second one being answered twice. A decoder
//! that resolved refs would need the object namespace, so it would either take
//! a reference to it — making every decode a borrow of live device state — or
//! resolve again later, which is the exact "resolve twice, get two answers"
//! shape the replacement exists to delete. So a record here carries the
//! guest's `u32` refs verbatim and the model resolves them once.
//!
//! # A refusal is typed, and the byte count is in it
//!
//! Every failure here is a wire fact: a record too short for its own body, an
//! opcode with no contract, a count that does not fit the record it is in.
//! Each one names what it saw, because a decode failure with no numbers in it
//! is a failure nobody can act on.

pub mod blit;
pub mod sync;

use crate::closure::Rail;

/// Why a record could not be lifted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeRefusal {
    /// The record is shorter than the body its opcode requires.
    Short {
        rail: Rail,
        opcode: u32,
        have: usize,
        need: usize,
    },
    /// The opcode belongs to no record this rail carries.
    ///
    /// Distinct from [`Self::Unjudged`]: this one is not in the rail's map at
    /// all, which is a stream that has gone wrong or a serializer this device
    /// has never seen. That is a different report from an opcode whose contract
    /// is merely open.
    UnknownOpcode { rail: Rail, opcode: u32 },
    /// The opcode is real and its contract is not established, so the model
    /// must not represent it.
    Unjudged { rail: Rail, opcode: u32 },
    /// The opcode's contract is established and the device refuses it.
    ///
    /// Distinct from [`Self::Unjudged`], which says nothing is known. This one
    /// says the row is settled and the settlement is a refusal, so a record
    /// that decoded perfectly must still not become an operation. Reporting the
    /// two the same way would make a deliberate refusal look like an open
    /// question and put it back on the work queue.
    RefusedByContract { rail: Rail, opcode: u32 },
    /// A counted array does not fit the record that declares it.
    ///
    /// The count is the guest's, so this is an ordinary hostile-input case and
    /// not a corrupt device. The declared count is reported beside the bytes
    /// available, because "the guest asked for 200" and "the record held 12" is
    /// the pair that identifies which of the two is wrong.
    CountOverruns {
        rail: Rail,
        opcode: u32,
        count: u32,
        have: usize,
    },
}

impl DecodeRefusal {
    /// The stable reason string for the failure channel.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::Short { .. } => "decode_record_short",
            Self::UnknownOpcode { .. } => "decode_opcode_unknown",
            Self::Unjudged { .. } => "decode_opcode_unjudged",
            Self::RefusedByContract { .. } => "decode_opcode_refused_by_contract",
            Self::CountOverruns { .. } => "decode_count_overruns_record",
        }
    }

    #[must_use]
    pub const fn rail(self) -> Rail {
        match self {
            Self::Short { rail, .. }
            | Self::UnknownOpcode { rail, .. }
            | Self::Unjudged { rail, .. }
            | Self::RefusedByContract { rail, .. }
            | Self::CountOverruns { rail, .. } => rail,
        }
    }

    #[must_use]
    pub const fn opcode(self) -> u32 {
        match self {
            Self::Short { opcode, .. }
            | Self::UnknownOpcode { opcode, .. }
            | Self::Unjudged { opcode, .. }
            | Self::RefusedByContract { opcode, .. }
            | Self::CountOverruns { opcode, .. } => opcode,
        }
    }
}

/// The refusal for an opcode this rail lifts no record for.
///
/// Three answers, and the difference is what a reader needs. An opcode the
/// ledger settled as [`crate::closure::Closure::Refused`] is refused *by
/// contract*: nothing is missing and nothing is to be built. One the ledger has
/// a row for but has not settled is unjudged, and the row says what is not yet
/// known. One with no row at all is a stream that has gone wrong, or a
/// serializer this device has never seen.
///
/// Collapsing the first two is the mistake worth naming: it would put a
/// deliberate refusal back on the work queue every time someone read the logs,
/// and it would let a genuinely open contract hide behind "we meant to do
/// that".
pub(crate) fn no_record(rail: Rail, opcode: u32) -> DecodeRefusal {
    match crate::closure::find(rail, opcode).map(|row| row.closure) {
        Some(crate::closure::Closure::Refused { .. }) => {
            DecodeRefusal::RefusedByContract { rail, opcode }
        }
        Some(_) => DecodeRefusal::Unjudged { rail, opcode },
        None => DecodeRefusal::UnknownOpcode { rail, opcode },
    }
}

/// Map a wire view error onto this layer's refusal.
///
/// The wire crate's error says a view did not fit; this layer's says which
/// record on which rail. Both facts are needed and neither is the other's.
pub(crate) fn short(rail: Rail, opcode: u32, have: usize, need: usize) -> DecodeRefusal {
    DecodeRefusal::Short {
        rail,
        opcode,
        have,
        need,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every refusal reason is distinct, so a log line says which shape was
    /// seen.
    #[test]
    fn refusal_reasons_are_distinct() {
        let all = [
            DecodeRefusal::Short {
                rail: Rail::Blit,
                opcode: 1,
                have: 0,
                need: 1,
            },
            DecodeRefusal::UnknownOpcode {
                rail: Rail::Blit,
                opcode: 1,
            },
            DecodeRefusal::Unjudged {
                rail: Rail::Blit,
                opcode: 1,
            },
            DecodeRefusal::RefusedByContract {
                rail: Rail::Blit,
                opcode: 1,
            },
            DecodeRefusal::CountOverruns {
                rail: Rail::Blit,
                opcode: 1,
                count: 2,
                have: 3,
            },
        ];
        let mut seen: alloc::vec::Vec<&str> = all.iter().map(|r| r.reason()).collect();
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(seen.len(), before);
        for refusal in all {
            assert_eq!(refusal.rail(), Rail::Blit);
            assert_eq!(refusal.opcode(), 1);
        }
    }
}
