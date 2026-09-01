//! The transaction envelope: what every accepted packet is, before anything
//! knows how a host would execute it.
//!
//! # One shape for GPU and non-GPU work
//!
//! A draw is not the unit of scheduling here and neither is an EXEC. Every
//! accepted FIFO packet is an ordered device transaction with the same
//! envelope: channel identity, a channel-local sequence, an ingress ordinal,
//! explicit prerequisites, a completion obligation, and one typed payload. A
//! resource delete, a display present and an EXEC differ in their payload and
//! in nothing else, because ordering and publication are owed to all three
//! equally — and the architecture this replaces gave each of them its own
//! partial mechanism.
//!
//! # Five payloads, and no catch-all among them
//!
//! [`PayloadClass::Control`] is the one that could rot into a bucket, so the
//! rule is written into [`classify`] and checked: a command may be `Control`
//! only when its established contract is a real control operation or an
//! acknowledged no-op. A command whose contract is *unknown* has no class at
//! all — it is a typed refusal at ingress, not a `Control` that quietly does
//! nothing — and [`crate::identity`]'s ordering guarantees never apply to work
//! that was never accepted.
//!
//! That rule is not enforceable by reading this file, so it is enforced against
//! the closure ledger: every packet class the ledger has judged maps to exactly
//! one payload, and every class it has *not* judged maps to none.

use reims_vgpu_protocol::closure::Closure;
use reims_vgpu_protocol::packets::{find, Channel};

/// What kind of work a transaction carries.
///
/// The payload contents are each their own type; this is the discriminant, and
/// it exists separately because ingress has to know which payload a packet
/// becomes before it has decoded one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PayloadClass {
    /// The GPU-work transaction: a counted resource table and an ordered list
    /// of serialized command streams.
    Exec,
    /// Task, object, resource and heap lifetime: create, delete, map, unmap,
    /// replace-physical, invalidate, synchronize, discard.
    ResourceLifecycle,
    /// A question with a decoded reply destination. The guest blocks on the
    /// answer, so an unanswered one is a wrong answer rather than lost work —
    /// which is why a query is its own class and not a control command.
    Query,
    /// A display present, carrying the surface identity and the presentation
    /// contract.
    Present,
    /// Display, cursor and channel control, and the acknowledged no-ops.
    /// **Not** a bucket for commands whose contract is unknown; see the module
    /// docs.
    Control,
}

impl PayloadClass {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Exec => "exec",
            Self::ResourceLifecycle => "resource_lifecycle",
            Self::Query => "query",
            Self::Present => "present",
            Self::Control => "control",
        }
    }
}

/// The payload class a packet becomes, or `None` when the model refuses it.
///
/// `None` is not "unhandled". It is the answer for a command whose contract has
/// not been established, and the caller's obligation is a typed refusal at
/// ingress — because a transaction the model cannot describe must not be given
/// ordering and completion guarantees it cannot honour.
#[must_use]
pub fn classify(channel: Channel, opcode: u16) -> Option<PayloadClass> {
    use Channel::{Child, Root};
    use PayloadClass::{Control, Exec, Present, Query, ResourceLifecycle};
    let judged = find(channel, opcode)?;
    if judged.closure.blocks_cutover() {
        return None;
    }
    Some(match (channel, opcode) {
        // The GPU-work packet, and the only one.
        (Child, 0x37) => Exec,

        // Task, object-list, resource and mapping lifetime. Root and child are
        // two views of one flat opcode space, so the shared numbers appear
        // twice on purpose rather than through a fallthrough.
        (Root, 0x20) | (Child, 0x20) => ResourceLifecycle,
        (Root, 0x33) | (Child, 0x33) => ResourceLifecycle,
        (Root, 0x38) | (Child, 0x38) => ResourceLifecycle,
        (Child, 0x22 | 0x25 | 0x34 | 0x35 | 0x36 | 0x39 | 0x3c | 0x3e | 0x3f) => ResourceLifecycle,

        // Questions with reply destinations.
        (Root, 0x2d | 0x3a) => Query,
        (Child, 0x3b | 0x40) => Query,

        // The three present forms. Enumerated rather than written as the
        // range they happen to occupy: they are three named commands whose
        // trailers differ, and a range would quietly adopt a fourth number if
        // the dispatch table ever grew one between them.
        #[allow(clippy::manual_range_patterns)]
        (Child, 0x06 | 0x07 | 0x08) => Present,

        // Display registration, cursor, channel lifetime, and the fence with no
        // payload. Everything left that the ledger has judged.
        _ => Control,
    })
}

/// Whether a class is one this model executes as GPU work.
///
/// Separate from [`PayloadClass::Exec`] being the only such class today,
/// because the question a reader asks is "does this reach an executor", and
/// answering it by naming one variant is how a second one gets added without
/// the readers noticing.
#[must_use]
pub const fn reaches_an_executor(class: PayloadClass) -> bool {
    matches!(class, PayloadClass::Exec)
}

/// Whether the closure ledger records this class as doing nothing by contract.
///
/// A `Control` transaction is not automatically a no-op — a present is not, a
/// cursor move is not — so the answer comes from the ledger rather than from
/// the payload class.
#[must_use]
pub fn is_acknowledged_noop(channel: Channel, opcode: u16) -> bool {
    matches!(
        find(channel, opcode).map(|p| p.closure),
        Some(Closure::ProvenNoOp { .. })
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_protocol::packets::LEDGER;

    /// The claim the module docs make and cannot check by being read: the
    /// classification is total over everything the ledger has judged, and empty
    /// over everything it has not.
    #[test]
    fn every_judged_packet_class_becomes_exactly_one_payload() {
        for p in LEDGER {
            let class = classify(p.channel, p.opcode);
            if p.closure.blocks_cutover() {
                assert_eq!(
                    class,
                    None,
                    "{} {:#04x} has no established contract, so the model must \
                     refuse it at ingress rather than accept it as a payload \
                     that quietly does nothing",
                    p.channel.name(),
                    p.opcode
                );
            } else {
                assert!(
                    class.is_some(),
                    "{} {:#04x} is judged {} and reaches no payload class",
                    p.channel.name(),
                    p.opcode,
                    p.closure.name()
                );
            }
        }
    }

    /// A packet the ledger has never heard of is not a `Control`.
    #[test]
    fn an_unjudged_opcode_has_no_payload() {
        assert_eq!(classify(Channel::Root, 0x00), None);
        assert_eq!(classify(Channel::Child, 0x1d), None);
        assert_eq!(classify(Channel::Child, 0xffff), None);
    }

    /// The retired slots are the acknowledged no-ops, and they are the only
    /// `Control` transactions that are.
    #[test]
    fn the_acknowledged_noops_are_the_retired_slots() {
        let noops: Vec<_> = LEDGER
            .iter()
            .filter(|p| is_acknowledged_noop(p.channel, p.opcode))
            .map(|p| (p.channel, p.opcode))
            .collect();
        assert_eq!(noops.len(), 15, "the reference host's retired slots");
        for (ch, op) in noops {
            assert_eq!(classify(ch, op), Some(PayloadClass::Control));
        }
    }

    #[test]
    fn exactly_one_packet_class_reaches_an_executor() {
        let executed: Vec<_> = LEDGER
            .iter()
            .filter_map(|p| classify(p.channel, p.opcode).map(|c| (p, c)))
            .filter(|(_, c)| reaches_an_executor(*c))
            .map(|(p, _)| (p.channel, p.opcode, p.name))
            .collect();
        assert_eq!(
            executed,
            vec![(Channel::Child, 0x37, "CmdExecIndirect2")],
            "a second packet class reaching an executor is a real change and \
             not a table edit"
        );
    }

    /// Present is not lifecycle and lifecycle is not control. Spot-checked
    /// against the readings that are easiest to get backwards.
    #[test]
    fn the_classes_that_look_alike_are_told_apart() {
        assert_eq!(
            classify(Channel::Child, 0x25),
            Some(PayloadClass::ResourceLifecycle),
            "CmdDeleteResource retires an object-table entry"
        );
        assert_eq!(
            classify(Channel::Child, 0x08),
            Some(PayloadClass::Present),
            "CmdDisplaySwapMapping is a present, not display control"
        );
        assert_eq!(
            classify(Channel::Child, 0x40),
            Some(PayloadClass::Query),
            "the guest blocks on the heap-texture reply, so it is a query and \
             not a control command that happens to write memory"
        );
        assert_eq!(
            classify(Channel::Child, 0x1e),
            Some(PayloadClass::Control),
            "CmdNOP's whole obligation is retiring its stamps"
        );
    }
}
