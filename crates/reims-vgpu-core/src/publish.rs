//! Ordered guest publication: when a completion stamp becomes something the
//! guest may read.
//!
//! # Completion is not publication
//!
//! Work finishing and the guest being told it finished are two events, and the
//! gap between them is the whole point of this module. A transaction's content
//! versions, event signals and fence updates become visible when its work
//! completes — out of order, as fast as the host finishes them. Its completion
//! stamp does not: a stamp is the word the guest polls to decide that
//! everything up to that point in its channel is done, so publishing it while
//! an earlier position in the same channel is still outstanding would tell the
//! guest a lie it has no way to detect.
//!
//! So each publication domain is a FIFO, and completion may release only the
//! ready positions at its head.
//!
//! # An independent domain stays free
//!
//! Head-of-line blocking within a domain is the contract. Head-of-line
//! blocking *between* domains is a bug, and the way to not have that bug is to
//! not have a structure that could express it: there is no shared queue here,
//! no global head, and no ordering between domains at all. A domain whose head
//! is stuck holds its own finished positions and nothing else's, and
//! [`Publisher::blocked`] counts exactly that so the cost is visible rather
//! than inferred.
//!
//! # A position that will never finish must leave
//!
//! A refused, cancelled or reset position is still a position, and leaving it
//! at the head would stall its domain forever. [`Publisher::withdraw`] removes
//! it and releases whatever was queued behind it — which is the only way a
//! refusal is *quiet* about ordering while being loud on the failure channel.

use crate::identity::{ChannelId, ChannelSequence, CompletionStamp};
use std::collections::{HashMap, VecDeque};

/// What a released position publishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Release {
    pub sequence: ChannelSequence,
    /// The stamp the guest may now read, if this position carries one. A
    /// position with no stamp still holds the FIFO, because the positions
    /// behind it are ordered against it and not against its stamp.
    pub stamp: Option<CompletionStamp>,
}

/// Why a channel could not be retired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetireRefusal {
    /// Positions are still admitted and unreleased. Retiring would drop
    /// publication the guest is owed.
    LivePositions { outstanding: usize },
}

impl RetireRefusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::LivePositions { .. } => "publish_live_positions",
        }
    }
}

/// One publication domain's FIFO.
#[derive(Debug, Default)]
struct Domain {
    /// Admitted and not yet released, in admission order. This is the FIFO;
    /// the sequence values need not be contiguous, because a channel may
    /// refuse a packet without admitting a position for it.
    order: VecDeque<ChannelSequence>,
    /// Positions whose work has completed, waiting for the head to reach them.
    finished: HashMap<ChannelSequence, Option<CompletionStamp>>,
}

/// Ordered guest publication for every channel.
#[derive(Debug, Default)]
pub struct Publisher {
    domains: HashMap<ChannelId, Domain>,
    released: usize,
}

impl Publisher {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Take a position in a channel's publication order.
    ///
    /// # Panics
    ///
    /// If `sequence` is not greater than the last position admitted into this
    /// channel. Publication order *is* channel order; admitting out of it
    /// would make the FIFO a queue of whatever arrived rather than a
    /// statement about the guest's channel.
    pub fn admit(&mut self, domain: ChannelId, sequence: ChannelSequence) {
        let queue = self.domains.entry(domain).or_default();
        assert!(
            queue.order.back().is_none_or(|last| sequence > *last),
            "publication order is channel order; {sequence:?} was admitted after a later position"
        );
        queue.order.push_back(sequence);
    }

    /// Record that a position's work has completed, and return everything the
    /// channel may now publish, in order.
    ///
    /// Returns empty when the position is not at the head — that is the
    /// ordinary case for out-of-order completion and is not an error.
    ///
    /// # Panics
    ///
    /// If the position was never admitted, or has already completed.
    /// Completing twice would publish a stamp twice.
    pub fn complete(
        &mut self,
        domain: ChannelId,
        sequence: ChannelSequence,
        stamp: Option<CompletionStamp>,
    ) -> Vec<Release> {
        let queue = self
            .domains
            .get_mut(&domain)
            .expect("completing a position in a channel that has none");
        assert!(
            queue.order.contains(&sequence),
            "completing {sequence:?}, which is not an outstanding position"
        );
        assert!(
            queue.finished.insert(sequence, stamp).is_none(),
            "completing {sequence:?} twice"
        );
        let released = Self::drain(queue);
        self.released += released.len();
        released
    }

    /// Remove a position that will never publish, and release whatever was
    /// queued behind it.
    ///
    /// # Panics
    ///
    /// If the position was never admitted, or has already completed —
    /// withdrawing completed work would discard publication the guest is owed
    /// rather than one it never will be.
    pub fn withdraw(&mut self, domain: ChannelId, sequence: ChannelSequence) -> Vec<Release> {
        let queue = self
            .domains
            .get_mut(&domain)
            .expect("withdrawing a position from a channel that has none");
        assert!(
            !queue.finished.contains_key(&sequence),
            "withdrawing {sequence:?}, which has already completed"
        );
        let at = queue
            .order
            .iter()
            .position(|s| *s == sequence)
            .expect("withdrawing a position that is not outstanding");
        queue.order.remove(at);
        let released = Self::drain(queue);
        self.released += released.len();
        released
    }

    /// Release the finished prefix of a domain's FIFO.
    fn drain(queue: &mut Domain) -> Vec<Release> {
        let mut out = Vec::new();
        while let Some(head) = queue.order.front().copied() {
            let Some(stamp) = queue.finished.remove(&head) else {
                break;
            };
            queue.order.pop_front();
            out.push(Release {
                sequence: head,
                stamp,
            });
        }
        out
    }

    /// Positions admitted into a channel and not yet released.
    #[must_use]
    pub fn outstanding(&self, domain: ChannelId) -> usize {
        self.domains.get(&domain).map_or(0, |q| q.order.len())
    }

    /// The position a channel is waiting on, if it is waiting on one.
    #[must_use]
    pub fn head(&self, domain: ChannelId) -> Option<ChannelSequence> {
        self.domains
            .get(&domain)
            .and_then(|q| q.order.front().copied())
    }

    /// Positions that have finished and are held behind an unfinished head,
    /// per channel.
    ///
    /// The cost of ordered publication, measured rather than assumed. A number
    /// that grows without the corresponding head ever finishing is the shape
    /// of a stuck channel, and it is a channel's own number: nothing here can
    /// make one domain's head hold another domain's work.
    #[must_use]
    pub fn blocked(&self) -> Vec<(ChannelId, usize)> {
        let mut out: Vec<_> = self
            .domains
            .iter()
            .filter(|(_, q)| !q.finished.is_empty())
            .map(|(d, q)| (*d, q.finished.len()))
            .collect();
        out.sort_unstable();
        out
    }

    /// Positions released across every channel.
    #[must_use]
    pub const fn released(&self) -> usize {
        self.released
    }

    /// End a channel's publication lifetime.
    ///
    /// # Errors
    ///
    /// If the channel still holds unreleased positions. A later definition of
    /// the same channel starts at position one and must not join the former
    /// lifetime's queue, so the former lifetime has to be empty before it can
    /// end.
    pub fn retire(&mut self, domain: ChannelId) -> Result<(), RetireRefusal> {
        let outstanding = self.outstanding(domain);
        if outstanding > 0 {
            return Err(RetireRefusal::LivePositions { outstanding });
        }
        self.domains.remove(&domain);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{StampSlot, StampValue};

    fn stamp(value: u32) -> Option<CompletionStamp> {
        Some(CompletionStamp {
            slot: StampSlot(1),
            value: StampValue(value),
        })
    }

    fn seq(n: u64) -> ChannelSequence {
        ChannelSequence(n)
    }

    /// The rule: work may finish in any order, and the guest is told in one.
    #[test]
    fn a_position_that_finishes_early_publishes_when_its_predecessor_does() {
        let mut p = Publisher::new();
        for n in 1..=3 {
            p.admit(ChannelId(1), seq(n));
        }
        assert!(
            p.complete(ChannelId(1), seq(3), stamp(3)).is_empty(),
            "third finished first and published nothing"
        );
        assert!(p.complete(ChannelId(1), seq(2), stamp(2)).is_empty());
        assert_eq!(p.blocked(), vec![(ChannelId(1), 2)]);
        assert_eq!(
            p.complete(ChannelId(1), seq(1), stamp(1)),
            vec![
                Release {
                    sequence: seq(1),
                    stamp: stamp(1)
                },
                Release {
                    sequence: seq(2),
                    stamp: stamp(2)
                },
                Release {
                    sequence: seq(3),
                    stamp: stamp(3)
                },
            ],
            "the head finishing releases the whole ready prefix, in order"
        );
        assert_eq!(p.outstanding(ChannelId(1)), 0);
        assert!(p.blocked().is_empty());
    }

    /// The structural claim: one domain's head holds its own work and nothing
    /// else's.
    #[test]
    fn a_stuck_channel_does_not_hold_another_channels_publication() {
        let mut p = Publisher::new();
        p.admit(ChannelId(1), seq(1));
        p.admit(ChannelId(1), seq(2));
        p.admit(ChannelId(2), seq(1));
        assert!(p.complete(ChannelId(1), seq(2), stamp(2)).is_empty());
        assert_eq!(
            p.complete(ChannelId(2), seq(1), stamp(9)),
            vec![Release {
                sequence: seq(1),
                stamp: stamp(9)
            }],
            "channel two's head is its own"
        );
        assert_eq!(p.blocked(), vec![(ChannelId(1), 1)]);
        assert_eq!(p.head(ChannelId(1)), Some(seq(1)));
        assert_eq!(p.head(ChannelId(2)), None);
    }

    /// A refusal is loud on the failure channel and quiet about ordering.
    #[test]
    fn withdrawing_a_position_releases_what_was_queued_behind_it() {
        let mut p = Publisher::new();
        for n in 1..=3 {
            p.admit(ChannelId(1), seq(n));
        }
        assert!(p.complete(ChannelId(1), seq(2), stamp(2)).is_empty());
        assert!(p.complete(ChannelId(1), seq(3), stamp(3)).is_empty());
        assert_eq!(
            p.withdraw(ChannelId(1), seq(1)),
            vec![
                Release {
                    sequence: seq(2),
                    stamp: stamp(2)
                },
                Release {
                    sequence: seq(3),
                    stamp: stamp(3)
                },
            ],
            "a position that will never publish must not stall the ones behind it"
        );
        assert_eq!(p.released(), 2);
    }

    /// A position without a stamp is still a position.
    #[test]
    fn a_position_that_publishes_no_stamp_still_holds_the_order() {
        let mut p = Publisher::new();
        p.admit(ChannelId(1), seq(1));
        p.admit(ChannelId(1), seq(2));
        assert!(p.complete(ChannelId(1), seq(2), stamp(2)).is_empty());
        assert_eq!(
            p.complete(ChannelId(1), seq(1), None),
            vec![
                Release {
                    sequence: seq(1),
                    stamp: None
                },
                Release {
                    sequence: seq(2),
                    stamp: stamp(2)
                },
            ]
        );
    }

    /// Sequences need not be contiguous: a channel may refuse a packet without
    /// ever admitting a position for it.
    #[test]
    fn gaps_in_the_sequence_do_not_stall_the_head() {
        let mut p = Publisher::new();
        p.admit(ChannelId(1), seq(1));
        p.admit(ChannelId(1), seq(7));
        assert_eq!(p.complete(ChannelId(1), seq(1), None).len(), 1);
        assert_eq!(p.complete(ChannelId(1), seq(7), None).len(), 1);
    }

    #[test]
    fn a_channel_with_live_positions_cannot_end_its_lifetime() {
        let mut p = Publisher::new();
        p.admit(ChannelId(1), seq(1));
        assert_eq!(
            p.retire(ChannelId(1)),
            Err(RetireRefusal::LivePositions { outstanding: 1 })
        );
        p.complete(ChannelId(1), seq(1), None);
        assert_eq!(p.retire(ChannelId(1)), Ok(()));
        // And a later definition starts at position one without joining the
        // lifetime that just ended.
        p.admit(ChannelId(1), seq(1));
        assert_eq!(p.outstanding(ChannelId(1)), 1);
    }

    #[test]
    #[should_panic(expected = "publication order is channel order")]
    fn admitting_a_position_out_of_channel_order_is_a_contract_violation() {
        let mut p = Publisher::new();
        p.admit(ChannelId(1), seq(5));
        p.admit(ChannelId(1), seq(2));
    }

    #[test]
    #[should_panic(expected = "twice")]
    fn completing_a_position_twice_would_publish_its_stamp_twice() {
        let mut p = Publisher::new();
        p.admit(ChannelId(1), seq(1));
        p.admit(ChannelId(1), seq(2));
        p.complete(ChannelId(1), seq(2), None);
        p.complete(ChannelId(1), seq(2), None);
    }
}
