//! From a packet this device drained to a packet the semantic model admits.
//!
//! # What this is for
//!
//! [`crate::runtime::drain`] takes bytes out of a ring and produces a
//! [`drain::Packet`]: an opcode, a record array, a completion word and a
//! payload. [`reims_vgpu_core::session::SessionModel`] admits a
//! [`reims_vgpu_core::session::Packet`]: an ordering domain, a semantic
//! lifetime, typed stamps and a resolved [`Payload`]. Those are the two ends of
//! the cutover and **nothing joined them** — the model could describe every
//! packet this device receives and could be handed none of them.
//!
//! This is that join, and it is a pure function: no device state is read, no
//! guest memory is touched, nothing is mutated. Everything it cannot answer
//! from its arguments it returns as a named [`Gap`] rather than approximating,
//! so the set of classes that can cross today is a value a test can assert
//! rather than a claim a reader has to believe.
//!
//! # The gaps are the cutover's remaining work, stated
//!
//! One class crosses now. [`reims_vgpu_core::control::resolve`] is a function
//! of the packet's own bytes, so a control packet needs nothing this device has
//! not already put in the `drain::Packet`. The other four each name one input
//! the model needs and this function is not given:
//!
//! - **Exec** needs the object-list resolver and the access source that
//!   [`reims_vgpu_core::walk::exec`] walks a command stream with.
//! - **Resource lifetime** needs the object and mapping namespaces, and for
//!   `SetObjectList` the guest table itself.
//! - **Query** needs its reply destination *resolved* — a backing and a window,
//!   not the address the request names.
//! - **Present** needs the accesses its target mapping resolves to.
//!
//! None of those is missing decode work: this device resolves all four today.
//! What it does not have is a *generation-stamped* namespace to resolve them
//! into, which is the model's, and giving this function a half-resolved answer
//! would be the adapter between two semantic models that the replacement plan
//! forbids. So they are gaps, they are named, and they close when their owner
//! lands — not here.
//!
//! # Every packet carries a completion word
//!
//! [`reims_vgpu_core::session::Packet::completion`] is an `Option`, because a
//! model may have packets that publish nothing. **This interface does not**:
//! the drain writes the header's completion word into the channel's stamp slot
//! for every packet it processes, and a packet that does not advance the fence
//! repeats the slot's current value rather than leaving it alone. Repeating is
//! idempotent and a wait decided against a repeat is decided the same way it
//! was before, so "does not signal" and "signals the value already there" are
//! the same event on the wire. This bridge therefore always produces `Some`,
//! and `None` stays available for a model that has the other case.
//!
//! # The slot is the channel's and the value is the packet's
//!
//! A `drain::Packet` carries a completion *value* and no slot. The slot belongs
//! to the FIFO — the root's is slot 0 and a child's is read once per drain from
//! its register block — so it arrives here as an argument. Wait records carry
//! their own raw index, masked to a slot by
//! [`stamp_slot_index`], which is the one place that mask is applied.

use crate::model::{is_child_channel, stamp_slot_index};
use crate::runtime::drain;
use reims_vgpu_core::control;
use reims_vgpu_core::identity::{
    ChannelId, CompletionStamp, SessionGeneration, StampSlot, StampValue, StampWait,
};
use reims_vgpu_core::session::Packet;
use reims_vgpu_core::transaction::{classify, Payload, PayloadClass};
use reims_vgpu_protocol::packets::Channel;

/// Which FIFO a packet was drained from.
///
/// **One value, because the two questions have one answer.**
/// [`reims_vgpu_core::session::Packet`] carries both a [`Channel`] — which
/// dispatch table the opcode is read against — and a [`ChannelId`] — which
/// ordering domain the packet joins. They are not independent: this device
/// numbers the root FIFO 0 and its children 1..[`MAX_CHANNELS`], which is the
/// rule [`is_child_channel`] states and the rule
/// [`reims_vgpu_core::control::ControlOp::Channel`] hands back a domain under.
/// A bridge taking the pair separately could be given `Channel::Root` with a
/// child's domain, and the packet would then be classified against the root's
/// opcode table and ordered on a child's channel. Deriving the channel from the
/// id makes that unrepresentable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fifo(ChannelId);

impl Fifo {
    /// The device's own channel, numbered 0.
    pub const ROOT: Fifo = Fifo(ChannelId(0));

    /// A guest-defined channel, or `None` when the id names no FIFO this
    /// device has. The bound is [`is_child_channel`]'s and not restated here.
    #[must_use]
    pub fn child(channel_id: u32) -> Option<Fifo> {
        is_child_channel(channel_id).then_some(Fifo(ChannelId(channel_id)))
    }

    /// The ordering domain packets on this FIFO join.
    #[must_use]
    pub const fn domain(self) -> ChannelId {
        self.0
    }

    /// Which opcode table this FIFO's packets are read against.
    #[must_use]
    pub const fn channel(self) -> Channel {
        if self.0 .0 == 0 {
            Channel::Root
        } else {
            Channel::Child
        }
    }
}

/// An input the model needs that this bridge is not given.
///
/// Not a refusal: the guest's packet is well formed and this device answers it
/// today. Each variant names the one thing missing, so a suite can assert the
/// partition — which classes cross and which do not — instead of a reader
/// having to infer it from what the code happens to handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gap {
    /// The ledger has not closed this row, so [`classify`] gives it no class
    /// and the model may not claim it. Production answers it alone.
    Unresolved,
    /// The object-list resolver and the access source an EXEC's command stream
    /// is walked with.
    ExecResolution,
    /// The object and mapping namespaces a lifetime command's resources
    /// resolve in.
    Namespaces,
    /// The reply destination, resolved to a backing and a window of it.
    ReplyDestination,
    /// The accesses the presented mapping resolves to.
    MappingAccesses,
}

impl Gap {
    /// The name this reaches the failure channel under.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Unresolved => "ingress_row_unresolved",
            Self::ExecResolution => "ingress_needs_exec_resolution",
            Self::Namespaces => "ingress_needs_namespaces",
            Self::ReplyDestination => "ingress_needs_reply_destination",
            Self::MappingAccesses => "ingress_needs_mapping_accesses",
        }
    }
}

/// Why a drained packet did not become a model packet.
///
/// The two arms are different obligations and are deliberately not one type. A
/// [`Self::Gap`] is this device's own incompleteness and closes when its owner
/// lands; a [`Self::Refused`] is the guest's bytes not being the command its
/// opcode names, and closes never.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blocked {
    Gap {
        channel: Channel,
        opcode: u16,
        gap: Gap,
    },
    Refused(control::ResolveRefusal),
}

impl Blocked {
    /// The name this reaches the failure channel under.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Gap { gap, .. } => gap.slug(),
            Self::Refused(refusal) => refusal.slug(),
        }
    }

    /// The gap, for a caller deciding whether to fall back to the legacy path.
    #[must_use]
    pub const fn gap(self) -> Option<Gap> {
        match self {
            Self::Gap { gap, .. } => Some(gap),
            Self::Refused(_) => None,
        }
    }
}

/// Build the model packet one drained packet describes.
///
/// `session` is the semantic lifetime the packet was **read under**, which is
/// the reader's fact and not this function's — see
/// [`reims_vgpu_core::session::Packet::session`]. `completion_slot` is the
/// FIFO's stamp slot, already masked by whoever read it.
///
/// # Errors
///
/// [`Blocked::Gap`] for every class this bridge cannot yet build, and
/// [`Blocked::Refused`] when a control packet's bytes are not its command.
pub fn packet(
    fifo: Fifo,
    session: SessionGeneration,
    completion_slot: StampSlot,
    drained: &drain::Packet,
) -> Result<Packet, Blocked> {
    let channel = fifo.channel();
    let opcode = drained.opcode;
    let blocked = |gap| Blocked::Gap {
        channel,
        opcode,
        gap,
    };

    // Exhaustive over `PayloadClass`, so a sixth class is a compile error here
    // rather than a packet that quietly reaches whichever arm came last. The
    // gap each class names is stated once, in this match, and nowhere else.
    let payload = match classify(channel, opcode) {
        None => return Err(blocked(Gap::Unresolved)),
        Some(PayloadClass::Exec) => return Err(blocked(Gap::ExecResolution)),
        Some(PayloadClass::ResourceLifecycle) => return Err(blocked(Gap::Namespaces)),
        Some(PayloadClass::Query) => return Err(blocked(Gap::ReplyDestination)),
        Some(PayloadClass::Present) => return Err(blocked(Gap::MappingAccesses)),
        Some(PayloadClass::Control) => Payload::Control(
            control::resolve(channel, opcode, &drained.payload).map_err(Blocked::Refused)?,
        ),
    };

    Ok(Packet {
        channel,
        domain: fifo.domain(),
        session,
        opcode,
        stamp_waits: drained
            .stamp_waits
            .iter()
            .map(|wait| StampWait {
                slot: StampSlot(stamp_slot_index(wait.index)),
                value: StampValue(wait.value),
            })
            .collect(),
        completion: Some(CompletionStamp {
            slot: completion_slot,
            value: StampValue(drained.completion_stamp),
        }),
        payload,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    // The channel bound, read only by the test that sweeps it: `Fifo::child`
    // asks `is_child_channel` and never the constant behind it.
    use crate::model::MAX_CHANNELS;
    use reims_vgpu_protocol::packets::LEDGER;

    /// A payload long enough for every control command's own layout. Only the
    /// two channel-lifetime commands read one at all.
    const ROOMY: usize = 64;

    fn drained(opcode: u16) -> drain::Packet {
        drain::Packet {
            opcode,
            stamp_waits: Vec::new(),
            total_size: 0,
            completion_stamp: 0,
            payload: vec![0u8; ROOMY],
            next_head: 0,
        }
    }

    fn fifo_for(channel: Channel) -> Fifo {
        match channel {
            Channel::Root => Fifo::ROOT,
            Channel::Child => Fifo::child(1).expect("channel 1 is a child"),
        }
    }

    /// **The cutover ledger.** Every row the protocol crate judged, put through
    /// this bridge, and the answer asserted against the class the row has.
    ///
    /// A row that changes class, or a gap that closes without this being
    /// updated, fails here rather than becoming a silent change in what crosses
    /// to the model.
    #[test]
    fn every_ledger_row_either_crosses_or_names_what_it_needs() {
        let mut crossed = 0usize;
        let mut gapped = 0usize;
        for row in LEDGER {
            let fifo = fifo_for(row.channel);
            let outcome = packet(
                fifo,
                SessionGeneration::FIRST,
                StampSlot(0),
                &drained(row.opcode),
            );
            let expected = match classify(row.channel, row.opcode) {
                Some(PayloadClass::Control) => None,
                Some(PayloadClass::Exec) => Some(Gap::ExecResolution),
                Some(PayloadClass::ResourceLifecycle) => Some(Gap::Namespaces),
                Some(PayloadClass::Query) => Some(Gap::ReplyDestination),
                Some(PayloadClass::Present) => Some(Gap::MappingAccesses),
                None => Some(Gap::Unresolved),
            };
            match (expected, outcome) {
                (None, Ok(built)) => {
                    crossed += 1;
                    assert_eq!(built.opcode, row.opcode);
                    assert_eq!(built.channel, row.channel);
                    assert_eq!(built.domain, fifo.domain());
                    assert!(
                        matches!(built.payload, Payload::Control(_)),
                        "{} {:#04x} ({}) built a payload that is not its class",
                        row.channel.name(),
                        row.opcode,
                        row.name
                    );
                }
                (
                    Some(want),
                    Err(Blocked::Gap {
                        gap,
                        channel,
                        opcode,
                    }),
                ) => {
                    gapped += 1;
                    assert_eq!(
                        gap,
                        want,
                        "{} {:#04x} ({}) named the wrong missing input",
                        row.channel.name(),
                        row.opcode,
                        row.name
                    );
                    assert_eq!((channel, opcode), (row.channel, row.opcode));
                }
                (want, got) => panic!(
                    "{} {:#04x} ({}) expected {want:?} and got {got:?}",
                    row.channel.name(),
                    row.opcode,
                    row.name
                ),
            }
        }
        assert_eq!(
            crossed + gapped,
            LEDGER.len(),
            "every row is accounted for exactly once"
        );
        assert!(
            crossed > 0 && gapped > 0,
            "one side of the partition is empty, so the assertions above compared nothing: \
             {crossed} crossed, {gapped} gapped"
        );
    }

    /// The one class that crosses is exactly the one the module claims, and it
    /// is the whole of it.
    #[test]
    fn control_is_the_only_class_that_crosses_today() {
        let crossing: Vec<_> = LEDGER
            .iter()
            .filter(|row| {
                packet(
                    fifo_for(row.channel),
                    SessionGeneration::FIRST,
                    StampSlot(0),
                    &drained(row.opcode),
                )
                .is_ok()
            })
            .map(|row| (row.channel, row.opcode))
            .collect();
        let control: Vec<_> = LEDGER
            .iter()
            .filter(|row| classify(row.channel, row.opcode) == Some(PayloadClass::Control))
            .map(|row| (row.channel, row.opcode))
            .collect();
        assert_eq!(crossing, control);
        assert_eq!(
            control.len(),
            23,
            "the ledger's control rows changed; what crosses to the model is not what the \
             module documentation says it is"
        );
    }

    /// A channel-lifetime command with no room for its domain is refused, not
    /// defaulted. Opening domain 0 would name the root FIFO.
    #[test]
    fn a_channel_command_too_short_to_name_a_domain_is_refused() {
        let mut short = drained(0x30);
        short.payload.clear();
        let refusal = packet(Fifo::ROOT, SessionGeneration::FIRST, StampSlot(0), &short)
            .expect_err("no domain, no transition");
        assert!(matches!(refusal, Blocked::Refused(_)));
        assert_eq!(
            refusal.gap(),
            None,
            "a short payload is not a missing input"
        );
        assert_eq!(
            control::ControlKind::of(Channel::Root, 0x30)
                .and_then(control::ControlKind::channel_transition),
            Some(control::ChannelTransition::Open),
            "the opcode above stopped being the channel-open command, so this test refuses \
             something else"
        );
    }

    /// The envelope: the slot is the FIFO's, the value is the packet's, and a
    /// wait's raw index is masked exactly once.
    #[test]
    fn the_envelope_carries_the_channels_slot_and_the_packets_values() {
        let raw_index = u32::MAX;
        let mut nop = drained(0x1e);
        nop.completion_stamp = 0xDEAD_BEEF;
        nop.stamp_waits = vec![drain::StampWait {
            index: raw_index,
            value: 7,
        }];
        let built = packet(
            fifo_for(Channel::Child),
            SessionGeneration::FIRST,
            StampSlot(3),
            &nop,
        )
        .expect("CmdNOP is control");
        assert_eq!(
            built.completion,
            Some(CompletionStamp {
                slot: StampSlot(3),
                value: StampValue(0xDEAD_BEEF),
            })
        );
        assert_eq!(
            built.stamp_waits,
            vec![StampWait {
                slot: StampSlot(stamp_slot_index(raw_index)),
                value: StampValue(7),
            }]
        );
        assert_ne!(
            stamp_slot_index(raw_index),
            raw_index,
            "the raw index used above is not masked by anything, so the assertion that it was \
             masked would hold for a bridge that carried it through untouched"
        );
    }

    /// The device's channel numbering, which is what makes the channel and the
    /// domain one value rather than two that can disagree.
    #[test]
    fn the_root_fifo_is_channel_zero_and_children_are_the_rest() {
        assert_eq!(Fifo::ROOT.domain(), ChannelId(0));
        assert_eq!(Fifo::ROOT.channel(), Channel::Root);
        assert_eq!(
            Fifo::child(0),
            None,
            "channel 0 is the root FIFO, not a child"
        );
        assert_eq!(
            Fifo::child(MAX_CHANNELS as u32),
            None,
            "one past the last channel this device has"
        );
        for id in 1..MAX_CHANNELS as u32 {
            let fifo = Fifo::child(id).expect("a child this device has");
            assert_eq!(fifo.domain(), ChannelId(id));
            assert_eq!(fifo.channel(), Channel::Child);
        }
    }
}
