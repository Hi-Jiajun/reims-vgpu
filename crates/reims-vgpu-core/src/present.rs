//! One guest surface's presentation stream: what is in flight, in what order,
//! and what the old swapchain is still owed.
//!
//! # The returned image count is authoritative and the requested depth is not
//!
//! A device asks a host for a buffering depth and the host returns however many
//! images it decided to give. Every bound that matters — how many frames may be
//! in flight, when acquire has to wait — is the *returned* count. Treating the
//! request as the bound is the bug where a host that returned four images is
//! driven as though it had three, or worse as though it had five, and the
//! symptom is a stall or a validation error a long way from here. So the
//! requested depth is recorded as policy and never compared against anything.
//!
//! # Backpressure stops presentation and nothing else
//!
//! [`PresentStream::acquire`] never blocks. When every returned image is in
//! flight it refuses, and the refusal is this stream's alone: no draw, no
//! resource lifecycle and no other surface is held by it. A device that waited
//! for an image inside a general drain would let a slow display stop compute.
//!
//! # Order is FIFO unless the contract says a pending present may be replaced
//!
//! Presenting in submission order is the default and the only safe assumption.
//! Some presentation contracts allow a newer frame to supersede one that has
//! not been shown yet, which is a real latency win and a real correctness
//! hazard: superseding a frame the guest was told was presented is a dropped
//! frame the guest cannot account for. [`Order`] is therefore a value this
//! stream is constructed with, chosen from a structural capability by the layer
//! that has one, and never inferred here.
//!
//! # A replaced swapchain is retired, not destroyed
//!
//! Resizing or reconfiguring produces a new swapchain generation while the old
//! one's images are still being read by submitted work. The old generation goes
//! to deferred retirement against the timeline point of its last use — the same
//! exactness [`crate::retire`] applies to every other native object — and the
//! caller gets it back when that point is reached, never before.

use crate::identity::TimelinePoint;
use std::collections::VecDeque;
use std::num::NonZeroU64;

/// One configuration of a surface's swapchain.
///
/// Changes on every reconfiguration, so a ticket from before a resize cannot be
/// completed against the swapchain that replaced it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SwapchainGeneration(NonZeroU64);

impl SwapchainGeneration {
    pub const FIRST: SwapchainGeneration = SwapchainGeneration(NonZeroU64::MIN);

    #[must_use]
    pub const fn next(self) -> SwapchainGeneration {
        match NonZeroU64::new(self.0.get().saturating_add(1)) {
            Some(n) => SwapchainGeneration(n),
            None => SwapchainGeneration::FIRST,
        }
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Whether a pending present may be replaced by a newer one.
///
/// Not a guess and not a device name: the layer that knows the presentation
/// contract passes it in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Every acquired frame is presented, in the order it was acquired.
    Fifo,
    /// A frame that has not been shown yet may be superseded by a newer one.
    Superseding,
}

/// A frame in flight, and where it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// An image has been reserved and the frame is not yet drawn.
    AcquirePending,
    /// The frame is drawn and may be queued.
    Ready,
    /// Handed to the host's presentation queue.
    Queued,
}

/// One frame's place in the stream.
///
/// Not `Clone`: two copies of a ticket are two claims on one image, and the
/// second one to be completed would release an image the first still holds.
#[derive(Debug, PartialEq, Eq)]
pub struct Ticket {
    pub generation: SwapchainGeneration,
    /// Position in this generation's presentation order.
    pub sequence: u64,
    /// Which of the returned images this frame occupies.
    pub image: usize,
}

/// Why the stream would not do something.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Every returned image is in flight. Presentation waits; nothing else
    /// does.
    NoFreeImage { images: usize },
    /// The ticket belongs to a swapchain that has been replaced.
    StaleGeneration {
        named: SwapchainGeneration,
        current: SwapchainGeneration,
    },
    /// The frame is not in the phase this transition starts from.
    WrongPhase { at: Phase, expected: Phase },
    /// A frame was asked to be presented ahead of an earlier one under an
    /// order that does not permit it.
    OutOfOrder { head: u64, named: u64 },
    /// A frame was superseded by a newer one, under an order that permits it.
    /// Not an error: the caller owes the guest this fact and owes the host
    /// nothing.
    Superseded { by: u64 },
    /// No swapchain has been configured, so there is nothing to acquire from.
    NotConfigured,
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NoFreeImage { .. } => "present_no_free_image",
            Self::StaleGeneration { .. } => "present_stale_generation",
            Self::WrongPhase { .. } => "present_wrong_phase",
            Self::OutOfOrder { .. } => "present_out_of_order",
            Self::Superseded { .. } => "present_superseded",
            Self::NotConfigured => "present_not_configured",
        }
    }
}

/// A swapchain whose images are still being read by submitted work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "a retired swapchain nothing retires is a leak"]
pub struct Retired {
    pub generation: SwapchainGeneration,
    /// The point after which nothing reads its images.
    pub last_use: TimelinePoint,
}

#[derive(Debug)]
struct InFlight {
    sequence: u64,
    image: usize,
    phase: Phase,
}

/// One guest surface's presentation stream.
#[derive(Debug)]
pub struct PresentStream {
    order: Order,
    generation: SwapchainGeneration,
    /// What the host actually returned. The only bound anything here uses.
    images: Option<usize>,
    /// What was asked for. Policy, recorded so a report can say what the host
    /// did with it, and never compared against anything.
    requested_depth: Option<usize>,
    next_sequence: u64,
    in_flight: VecDeque<InFlight>,
    retiring: Vec<Retired>,
    presented: usize,
    superseded: usize,
    backpressured: usize,
}

impl PresentStream {
    #[must_use]
    pub fn new(order: Order) -> Self {
        Self {
            order,
            generation: SwapchainGeneration::FIRST,
            images: None,
            requested_depth: None,
            next_sequence: 0,
            in_flight: VecDeque::new(),
            retiring: Vec::new(),
            presented: 0,
            superseded: 0,
            backpressured: 0,
        }
    }

    #[must_use]
    pub const fn order(&self) -> Order {
        self.order
    }

    #[must_use]
    pub const fn generation(&self) -> SwapchainGeneration {
        self.generation
    }

    /// The image count the host returned, which is the only bound this stream
    /// uses.
    #[must_use]
    pub const fn images(&self) -> Option<usize> {
        self.images
    }

    /// The depth that was requested. Policy; nothing here compares against it.
    #[must_use]
    pub const fn requested_depth(&self) -> Option<usize> {
        self.requested_depth
    }

    /// Configure the first swapchain.
    ///
    /// # Panics
    ///
    /// If the host returned no images. A swapchain of zero images cannot
    /// present, and admitting one would turn every acquire into backpressure
    /// that never lifts.
    pub fn configure(&mut self, requested_depth: usize, returned_images: usize) {
        assert!(
            returned_images > 0,
            "a swapchain of no images cannot present"
        );
        self.requested_depth = Some(requested_depth);
        self.images = Some(returned_images);
    }

    /// Replace the swapchain, handing the old generation to deferred
    /// retirement.
    ///
    /// Frames in flight on the old generation are dropped from this stream's
    /// order — they belong to a swapchain that no longer exists — but the
    /// swapchain itself is not destroyed: `last_use` is when its images stop
    /// being read, and the caller gets it back from [`Self::reached`] then.
    ///
    /// # Panics
    ///
    /// As [`Self::configure`].
    pub fn replace(
        &mut self,
        requested_depth: usize,
        returned_images: usize,
        last_use: TimelinePoint,
    ) -> Retired {
        let retired = Retired {
            generation: self.generation,
            last_use,
        };
        self.retiring.push(retired);
        self.generation = self.generation.next();
        self.in_flight.clear();
        self.next_sequence = 0;
        self.configure(requested_depth, returned_images);
        retired
    }

    /// The timeline reached `at`: take the swapchains nothing reads any more.
    pub fn reached(&mut self, at: TimelinePoint) -> Vec<Retired> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.retiring.len() {
            if at.reached(self.retiring[i].last_use) {
                out.push(self.retiring.swap_remove(i));
            } else {
                i += 1;
            }
        }
        out.sort_unstable_by_key(|r| r.generation);
        out
    }

    /// Reserve an image for the next frame.
    ///
    /// Never waits. When every returned image is in flight this refuses, and
    /// the refusal holds up presentation and nothing else.
    ///
    /// # Errors
    ///
    /// [`Refusal::NotConfigured`] before a swapchain exists, and
    /// [`Refusal::NoFreeImage`] when all of them are in flight.
    pub fn acquire(&mut self) -> Result<Ticket, Refusal> {
        let images = self.images.ok_or(Refusal::NotConfigured)?;
        if self.in_flight.len() >= images {
            self.backpressured += 1;
            return Err(Refusal::NoFreeImage { images });
        }
        let used: Vec<usize> = self.in_flight.iter().map(|f| f.image).collect();
        let image = (0..images)
            .find(|i| !used.contains(i))
            .expect("fewer frames in flight than images");
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        self.in_flight.push_back(InFlight {
            sequence,
            image,
            phase: Phase::AcquirePending,
        });
        Ok(Ticket {
            generation: self.generation,
            sequence,
            image,
        })
    }

    /// Where a ticket's frame is.
    #[must_use]
    pub fn phase(&self, ticket: &Ticket) -> Option<Phase> {
        if ticket.generation != self.generation {
            return None;
        }
        self.in_flight
            .iter()
            .find(|f| f.sequence == ticket.sequence)
            .map(|f| f.phase)
    }

    /// The frame is drawn.
    ///
    /// # Errors
    ///
    /// If the ticket is from a replaced swapchain, or the frame is not
    /// awaiting its content.
    pub fn ready(&mut self, ticket: &Ticket) -> Result<(), Refusal> {
        self.advance(ticket, Phase::AcquirePending, Phase::Ready)
    }

    /// The frame has been handed to the host's presentation queue.
    ///
    /// Under [`Order::Fifo`] a frame may only be queued when it is at the head
    /// of the order: presenting out of submission order is a reordered frame
    /// the guest cannot account for.
    ///
    /// Under [`Order::Superseding`] a newer frame may go first, and every
    /// earlier one still in flight is dropped and reported — the caller owes
    /// the guest that fact.
    ///
    /// # Errors
    ///
    /// As [`Self::ready`], plus [`Refusal::OutOfOrder`] under FIFO.
    pub fn queue(&mut self, ticket: &Ticket) -> Result<Vec<u64>, Refusal> {
        self.check_generation(ticket)?;
        let head = self
            .in_flight
            .front()
            .map_or(ticket.sequence, |f| f.sequence);
        let mut dropped = Vec::new();
        if head != ticket.sequence {
            match self.order {
                Order::Fifo => {
                    return Err(Refusal::OutOfOrder {
                        head,
                        named: ticket.sequence,
                    })
                }
                Order::Superseding => {
                    while let Some(front) = self.in_flight.front() {
                        if front.sequence >= ticket.sequence {
                            break;
                        }
                        dropped.push(front.sequence);
                        self.in_flight.pop_front();
                        self.superseded += 1;
                    }
                }
            }
        }
        self.advance(ticket, Phase::Ready, Phase::Queued)?;
        Ok(dropped)
    }

    /// The host has shown the frame; its image is free again.
    ///
    /// # Errors
    ///
    /// As [`Self::ready`].
    pub fn complete(&mut self, ticket: &Ticket) -> Result<(), Refusal> {
        self.check_generation(ticket)?;
        let Some(at) = self
            .in_flight
            .iter()
            .position(|f| f.sequence == ticket.sequence)
        else {
            return Err(Refusal::Superseded {
                by: self.next_sequence,
            });
        };
        if self.in_flight[at].phase != Phase::Queued {
            return Err(Refusal::WrongPhase {
                at: self.in_flight[at].phase,
                expected: Phase::Queued,
            });
        }
        self.in_flight.remove(at);
        self.presented += 1;
        Ok(())
    }

    fn check_generation(&self, ticket: &Ticket) -> Result<(), Refusal> {
        if ticket.generation == self.generation {
            return Ok(());
        }
        Err(Refusal::StaleGeneration {
            named: ticket.generation,
            current: self.generation,
        })
    }

    fn advance(&mut self, ticket: &Ticket, from: Phase, to: Phase) -> Result<(), Refusal> {
        self.check_generation(ticket)?;
        let Some(frame) = self
            .in_flight
            .iter_mut()
            .find(|f| f.sequence == ticket.sequence)
        else {
            return Err(Refusal::Superseded {
                by: self.next_sequence,
            });
        };
        if frame.phase != from {
            return Err(Refusal::WrongPhase {
                at: frame.phase,
                expected: from,
            });
        }
        frame.phase = to;
        Ok(())
    }

    /// Frames occupying an image.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.in_flight.len()
    }

    /// Swapchains handed to deferred retirement and not yet reached.
    #[must_use]
    pub fn awaiting_retirement(&self) -> usize {
        self.retiring.len()
    }

    /// Frames presented, frames superseded, and acquires that found no free
    /// image.
    ///
    /// The third number is what says whether the host's returned count is the
    /// bottleneck, which is not something a requested depth could answer.
    #[must_use]
    pub const fn census(&self) -> (usize, usize, usize) {
        (self.presented, self.superseded, self.backpressured)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(n: u64) -> TimelinePoint {
        TimelinePoint(n)
    }

    fn present(stream: &mut PresentStream) -> Ticket {
        let ticket = stream.acquire().expect("an image");
        stream.ready(&ticket).expect("acquired");
        stream.queue(&ticket).expect("at the head");
        ticket
    }

    /// The count that bounds anything is the one the host returned.
    #[test]
    fn the_returned_image_count_bounds_the_stream_and_the_request_does_not() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(3, 2);
        assert_eq!(s.requested_depth(), Some(3));
        assert_eq!(s.images(), Some(2));

        let first = s.acquire().expect("first of two");
        let second = s.acquire().expect("second of two");
        assert_ne!(first.image, second.image, "two frames, two images");
        assert_eq!(
            s.acquire(),
            Err(Refusal::NoFreeImage { images: 2 }),
            "three were asked for and two were given; two is the bound"
        );
        assert_eq!(s.census().2, 1);
    }

    #[test]
    fn an_unconfigured_stream_has_nothing_to_acquire_from() {
        let mut s = PresentStream::new(Order::Fifo);
        assert_eq!(s.acquire(), Err(Refusal::NotConfigured));
    }

    /// Completing a frame frees its image, and the freed image is reused.
    #[test]
    fn a_presented_frame_returns_its_image_to_the_stream() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(2, 2);
        let first = present(&mut s);
        let image = first.image;
        s.complete(&first).expect("queued");
        assert_eq!(s.in_flight(), 0, "the frame is shown and holds nothing");

        let a = s.acquire().expect("free");
        let b = s.acquire().expect("free");
        assert!([a.image, b.image].contains(&image), "the image came back");
        assert_eq!(s.census().0, 1);
    }

    /// The phases are a machine, not a set of flags.
    #[test]
    fn a_frame_cannot_be_queued_before_it_is_drawn() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(1, 1);
        let ticket = s.acquire().expect("an image");
        assert_eq!(s.phase(&ticket), Some(Phase::AcquirePending));
        assert_eq!(
            s.queue(&ticket),
            Err(Refusal::WrongPhase {
                at: Phase::AcquirePending,
                expected: Phase::Ready
            })
        );
        s.ready(&ticket).expect("acquired");
        assert_eq!(
            s.ready(&ticket),
            Err(Refusal::WrongPhase {
                at: Phase::Ready,
                expected: Phase::AcquirePending
            }),
            "and it cannot be drawn twice"
        );
        s.queue(&ticket).expect("ready");
        assert_eq!(s.phase(&ticket), Some(Phase::Queued));
    }

    #[test]
    fn a_frame_cannot_be_completed_before_it_is_queued() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(1, 1);
        let ticket = s.acquire().expect("an image");
        assert_eq!(
            s.complete(&ticket),
            Err(Refusal::WrongPhase {
                at: Phase::AcquirePending,
                expected: Phase::Queued
            })
        );
    }

    /// FIFO means what it says: the head goes first.
    #[test]
    fn fifo_refuses_to_present_a_later_frame_first() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(2, 2);
        let first = s.acquire().expect("an image");
        let second = s.acquire().expect("an image");
        s.ready(&second).expect("acquired");
        assert_eq!(
            s.queue(&second),
            Err(Refusal::OutOfOrder {
                head: first.sequence,
                named: second.sequence
            }),
            "a reordered frame is one the guest cannot account for"
        );
    }

    /// And a superseding contract drops the frames it passed, by name.
    #[test]
    fn a_superseding_stream_reports_every_frame_it_dropped() {
        let mut s = PresentStream::new(Order::Superseding);
        s.configure(3, 3);
        let first = s.acquire().expect("an image");
        let second = s.acquire().expect("an image");
        let third = s.acquire().expect("an image");
        s.ready(&third).expect("acquired");
        assert_eq!(
            s.queue(&third).expect("superseding"),
            vec![first.sequence, second.sequence],
            "the caller owes the guest these two"
        );
        assert_eq!(s.census().1, 2);
        assert_eq!(
            s.complete(&first),
            Err(Refusal::Superseded { by: 3 }),
            "a dropped frame does not complete"
        );
        s.complete(&third).expect("queued");
        assert_eq!(s.in_flight(), 0);
    }

    /// A replaced swapchain is retired against a timeline point, not destroyed.
    #[test]
    fn replacing_a_swapchain_defers_it_to_its_last_use() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(2, 2);
        let old = present(&mut s);

        let retired = s.replace(2, 3, at(40));
        assert_eq!(retired.generation, SwapchainGeneration::FIRST);
        assert_eq!(s.generation(), SwapchainGeneration::FIRST.next());
        assert_eq!(s.images(), Some(3));
        assert_eq!(s.in_flight(), 0, "frames of a gone swapchain are gone");

        assert_eq!(
            s.complete(&old),
            Err(Refusal::StaleGeneration {
                named: SwapchainGeneration::FIRST,
                current: SwapchainGeneration::FIRST.next(),
            }),
            "a ticket from before the resize names a swapchain that is gone"
        );

        assert!(
            s.reached(at(39)).is_empty(),
            "submitted work is still reading its images"
        );
        assert_eq!(s.awaiting_retirement(), 1);
        assert_eq!(s.reached(at(40)), vec![retired]);
        assert_eq!(s.awaiting_retirement(), 0);
    }

    /// Two replacements in flight retire independently and in generation order.
    #[test]
    fn several_retired_swapchains_come_back_in_order() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(2, 2);
        let first = s.replace(2, 2, at(10));
        let second = s.replace(2, 2, at(5));
        assert_eq!(s.reached(at(7)), vec![second]);
        assert_eq!(s.reached(at(10)), vec![first]);
    }

    /// Backpressure is this stream's and nothing else's: the refusal names the
    /// bound and nothing here can hold another service.
    #[test]
    fn backpressure_is_a_refusal_and_never_a_wait() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(1, 1);
        let held = s.acquire().expect("an image");
        for _ in 0..8 {
            assert_eq!(s.acquire(), Err(Refusal::NoFreeImage { images: 1 }));
        }
        assert_eq!(s.census().2, 8, "eight refusals and not one wait");
        s.ready(&held).expect("acquired");
        s.queue(&held).expect("at the head");
        s.complete(&held).expect("queued");
        s.acquire().expect("the image came back");
    }

    #[test]
    #[should_panic(expected = "cannot present")]
    fn a_swapchain_of_no_images_is_a_contract_violation() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(2, 0);
    }
}
