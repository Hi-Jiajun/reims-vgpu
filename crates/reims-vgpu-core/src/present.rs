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
//! # A present that cannot acquire is parked here, and nowhere else
//!
//! Refusing an acquire is not the whole answer, because the packet that asked
//! for it is a transaction the guest is waiting on: dropping it is a lost frame
//! and a completion word that never arrives. So a present with no image to take
//! parks in *this surface's* queue, in arrival order, and
//! [`PresentStream::wake`] hands it back when an image frees. Nothing outside
//! the surface is held: another surface has its own stream, and a channel with
//! no present on it never touches this queue at all. That is what
//! "backpressure does not head-of-line block" means structurally rather than by
//! convention — there is no shared queue for a slow display to fill.
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

use crate::identity::{CompletionStamp, IngressOrdinal, MappingId, TaskId, TimelinePoint};
use reims_vgpu_protocol::packets::Channel;
pub use reims_vgpu_protocol::present::PresentForm;
use std::collections::VecDeque;
use std::num::NonZeroU64;

/// One present packet, decoded.
///
/// What the guest asked to show, and — for the two forms that name one — the
/// task that owns it. The pipe or display index at word zero is deliberately
/// absent: no path in this device reads it, and a field nothing reads is one
/// that quietly acquires a wrong meaning. It stays available on
/// [`reims_vgpu_protocol::present::Trailer`] for the layer that has a display.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentPacket {
    pub form: PresentForm,
    /// What to show. See [`MappingId`] for why this is not an object-list ref
    /// even though both arrive as `u32`.
    pub mapping: MappingId,
    /// The task that owns what is being shown, for the two forms whose trailer
    /// names one. `None` for [`PresentForm::SwapMapping`], whose word at that
    /// position is unidentified — reading it as a task would report whatever it
    /// happens to hold.
    pub task: Option<TaskId>,
}

/// Why a present packet's bytes did not become one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveRefusal {
    /// The packet is not a present.
    NotPresent { channel: Channel, opcode: u16 },
    /// The payload is shorter than the emitting command's own trailer, so
    /// there is nothing to show. Refused rather than clamped: presenting
    /// mapping zero and completing the packet in silence is a frame the guest
    /// believes it showed.
    Payload(reims_vgpu_protocol::present::Refusal),
}

impl ResolveRefusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NotPresent { .. } => "present_not_a_present_packet",
            Self::Payload(inner) => inner.slug(),
        }
    }
}

/// Turn one present packet into what it asks to show.
///
/// **The join between the three trailers and the model.** The wire could read
/// a present's words and this crate could name which of the three forms a
/// packet is, and nothing carried a packet from one to the other — so
/// [`crate::transaction::Payload::Present`] named a form and lost the target.
///
/// Where the frame goes and what has to happen for it to arrive are
/// [`PresentStream`]'s and the display layer's. This says what the guest asked
/// for.
///
/// # Errors
///
/// [`ResolveRefusal`]: a packet that is not a present, or one too short to
/// carry its own trailer.
pub fn resolve(
    channel: Channel,
    opcode: u16,
    payload: &[u8],
) -> Result<PresentPacket, ResolveRefusal> {
    let Some(form) = PresentForm::of(channel, opcode) else {
        return Err(ResolveRefusal::NotPresent { channel, opcode });
    };
    let trailer =
        reims_vgpu_protocol::present::trailer(form, payload).map_err(ResolveRefusal::Payload)?;
    Ok(PresentPacket {
        form,
        mapping: MappingId(trailer.target),
        task: trailer.task.map(TaskId),
    })
}

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

/// A present packet that has not got an image yet.
///
/// Carries the completion word because the packet's obligation travels with it:
/// a parked present that lost its stamp is a guest waiting on a word nothing
/// can publish any more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentRequest {
    pub ingress: IngressOrdinal,
    pub stamp: Option<CompletionStamp>,
}

/// What admitting a present packet did.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "a present that is neither acquired nor parked is a frame nothing will ever show"]
pub enum Admission {
    /// An image was free and the frame may be drawn.
    Acquired {
        request: PresentRequest,
        ticket: Ticket,
    },
    /// Every returned image is in flight. The request is held in this surface's
    /// own queue behind `ahead` others, and nothing outside this surface is
    /// held with it.
    Parked { ahead: usize },
    /// No swapchain has been configured. Parking would be waiting for an image
    /// from a swapchain that does not exist, so this is the one present the
    /// caller has to answer for itself.
    NotConfigured,
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
    /// Present packets waiting for an image, oldest first.
    parked: VecDeque<PresentRequest>,
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
            parked: VecDeque::new(),
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
        let outcome = self.take_image();
        if matches!(outcome, Err(Refusal::NoFreeImage { .. })) {
            self.backpressured += 1;
        }
        outcome
    }

    /// Reserve an image without charging the backpressure census.
    ///
    /// Separate because [`Self::wake`] asks whether an image is free on every
    /// image release, and counting those as backpressure would make the number
    /// that says whether the returned count is the bottleneck grow with how
    /// often the stream is polled rather than with how often a frame waited.
    fn take_image(&mut self) -> Result<Ticket, Refusal> {
        let images = self.images.ok_or(Refusal::NotConfigured)?;
        if self.in_flight.len() >= images {
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

    /// Admit a present packet: give it an image, or park it for one.
    ///
    /// A request parks behind anything already parked even when an image is
    /// free, because the frame ahead of it asked first and presenting them out
    /// of arrival order is the reordering [`Order`] exists to decide about —
    /// not something a queue drains its way into.
    pub fn submit(&mut self, request: PresentRequest) -> Admission {
        if self.parked.is_empty() {
            match self.take_image() {
                Ok(ticket) => return Admission::Acquired { request, ticket },
                Err(Refusal::NotConfigured) => return Admission::NotConfigured,
                Err(_) => self.backpressured += 1,
            }
        }
        let ahead = self.parked.len();
        self.parked.push_back(request);
        Admission::Parked { ahead }
    }

    /// The parked presents that can acquire now, oldest first.
    ///
    /// Call after an image is released — [`Self::complete`] — or after a
    /// swapchain is replaced with more images. Returns nothing when nothing is
    /// waiting, which is the ordinary case and is not a refusal.
    #[must_use]
    pub fn wake(&mut self) -> Vec<(PresentRequest, Ticket)> {
        let mut woken = Vec::new();
        while !self.parked.is_empty() {
            let Ok(ticket) = self.take_image() else { break };
            let request = self.parked.pop_front().expect("not empty");
            woken.push((request, ticket));
        }
        woken
    }

    /// Presents waiting for an image.
    #[must_use]
    pub fn parked(&self) -> usize {
        self.parked.len()
    }

    /// Give back every parked present without acquiring for it.
    ///
    /// For a teardown or a reset: the frames will not be shown, and their
    /// completion words are still owed. Dropping them silently is the hang this
    /// returns them to prevent, so there is no path that discards a parked
    /// present without handing it to somebody.
    #[must_use = "a parked present nobody takes is a completion word nothing publishes"]
    pub fn abandon_parked(&mut self) -> Vec<PresentRequest> {
        self.parked.drain(..).collect()
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
    use crate::transaction::{classify, PayloadClass};
    use reims_vgpu_protocol::packets::LEDGER;

    /// A present payload whose target word sits where `form` puts it, with a
    /// different value in every other word — so a decoder reading the wrong
    /// slot cannot accidentally read the right number.
    fn present_bytes(form: PresentForm, mapping: u32, task: u32) -> Vec<u8> {
        let mut out = vec![0u8; form.trailer_len()];
        out[0..4].copy_from_slice(&0xDEADu32.to_le_bytes());
        let t = form.target_offset();
        out[t..t + 4].copy_from_slice(&mapping.to_le_bytes());
        if let Some(k) = form.task_offset() {
            out[k..k + 4].copy_from_slice(&task.to_le_bytes());
        }
        out
    }

    /// The join: a guest's present bytes become what it asked to show, on all
    /// three forms, with each form reading its own slots.
    #[test]
    fn a_present_payload_becomes_what_the_guest_asked_to_show() {
        for (channel, opcode, form) in [
            (Channel::Child, 0x06u16, PresentForm::Transaction2),
            (Channel::Child, 0x07, PresentForm::Transaction3),
            (Channel::Child, 0x08, PresentForm::SwapMapping),
        ] {
            let bytes = present_bytes(form, 0x5e, 0x77);
            assert_eq!(
                resolve(channel, opcode, &bytes),
                Ok(PresentPacket {
                    form,
                    mapping: MappingId(0x5e),
                    task: form.names_a_task().then_some(TaskId(0x77)),
                }),
                "{}",
                form.name()
            );
        }
    }

    /// The swap form's second word is unidentified, so nothing may report it as
    /// a task — and the two transaction forms keep theirs in *different* slots,
    /// so one read at the other's would be a plausible wrong answer.
    #[test]
    fn no_form_reads_another_forms_slots() {
        // op6's twelve bytes, padded to op7's thirty-six so the short refusal
        // is not what this test measures.
        let mut two = present_bytes(PresentForm::Transaction2, 0x5e, 0x77);
        two.resize(PresentForm::Transaction3.trailer_len(), 0);
        assert_eq!(
            resolve(Channel::Child, 0x07, &two).expect("long enough"),
            PresentPacket {
                form: PresentForm::Transaction3,
                mapping: MappingId(0x77),
                task: Some(TaskId(0x5e)),
            },
            "reading op6's payload as op7 must swap the two, not agree with it"
        );
        let swap = present_bytes(PresentForm::SwapMapping, 0x5e, 0);
        assert_eq!(
            resolve(Channel::Child, 0x08, &swap)
                .expect("long enough")
                .task,
            None
        );
    }

    /// Zero is a value the guest sends and it means nothing to show. A present
    /// carrying it is well formed, and its completion is owed in full.
    #[test]
    fn a_present_of_nothing_is_a_well_formed_present() {
        let bytes = present_bytes(PresentForm::SwapMapping, 0, 0);
        assert_eq!(
            resolve(Channel::Child, 0x08, &bytes)
                .expect("well formed")
                .mapping,
            MappingId(0)
        );
    }

    /// A payload too short to carry its own trailer is refused, not clamped:
    /// presenting mapping zero and completing in silence is a frame the guest
    /// believes it showed.
    #[test]
    fn a_present_too_short_for_its_trailer_is_refused() {
        for (opcode, form) in [
            (0x06u16, PresentForm::Transaction2),
            (0x07, PresentForm::Transaction3),
            (0x08, PresentForm::SwapMapping),
        ] {
            let need = form.trailer_len();
            let bytes = present_bytes(form, 0x5e, 0x77);
            let refusal = resolve(Channel::Child, opcode, &bytes[..need - 1])
                .expect_err("one byte under its trailer");
            assert_eq!(
                refusal,
                ResolveRefusal::Payload(reims_vgpu_protocol::present::Refusal::Short {
                    have: need - 1,
                    need,
                }),
                "{}",
                form.name()
            );
            assert_eq!(refusal.slug(), "display_present_short");
        }
    }

    /// And a packet that is not a present does not become one, whatever its
    /// payload holds.
    #[test]
    fn a_packet_that_is_not_a_present_does_not_resolve_to_one() {
        for p in LEDGER {
            if PresentForm::of(p.channel, p.opcode).is_some() {
                continue;
            }
            assert_eq!(
                resolve(p.channel, p.opcode, &[0u8; 64]),
                Err(ResolveRefusal::NotPresent {
                    channel: p.channel,
                    opcode: p.opcode
                }),
                "{} {:#04x} ({})",
                p.channel.name(),
                p.opcode,
                p.name
            );
        }
    }

    /// The claim every other payload class already makes about itself: the
    /// class's vocabulary is exhaustive over what the ledger judged into it,
    /// and empty over everything else. Present was the one class whose
    /// `every_judged_packet_reaches_a_class_and_a_meaning_within_it` arm was a
    /// hardcoded `true`, which is not a check.
    #[test]
    fn every_present_packet_has_exactly_one_form() {
        let mut seen: Vec<PresentForm> = Vec::new();
        for p in LEDGER {
            let form = PresentForm::of(p.channel, p.opcode);
            let is_present = classify(p.channel, p.opcode) == Some(PayloadClass::Present);
            assert_eq!(
                form.is_some(),
                is_present,
                "{} {:#04x} ({}) is classified {:?} and resolves to {:?}",
                p.channel.name(),
                p.opcode,
                p.name,
                classify(p.channel, p.opcode),
                form
            );
            if let Some(form) = form {
                assert!(
                    !seen.contains(&form),
                    "{} is two packets' form",
                    form.name()
                );
                seen.push(form);
            }
        }
        assert_eq!(seen.len(), 3, "the three present forms");
    }

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
    fn request(ingress: u64) -> PresentRequest {
        PresentRequest {
            ingress: IngressOrdinal(ingress),
            stamp: Some(CompletionStamp {
                slot: crate::identity::StampSlot(1),
                value: crate::identity::StampValue(ingress as u32),
            }),
        }
    }

    /// A present with no image left does not vanish and does not wait: it
    /// parks in this surface's own queue, and the image release is what hands
    /// it back.
    #[test]
    fn a_present_with_no_image_parks_and_wakes_in_arrival_order() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(2, 2);
        let mut tickets = Vec::new();
        for i in 0..2 {
            let Admission::Acquired { ticket, .. } = s.submit(request(i)) else {
                panic!("an image was free");
            };
            tickets.push(ticket);
        }
        assert_eq!(s.submit(request(2)), Admission::Parked { ahead: 0 });
        assert_eq!(s.submit(request(3)), Admission::Parked { ahead: 1 });
        assert_eq!(s.parked(), 2);
        assert!(s.wake().is_empty(), "no image has been released yet");

        s.ready(&tickets[0]).expect("drawn");
        s.queue(&tickets[0]).expect("head of the order");
        s.complete(&tickets[0]).expect("shown");
        let woken = s.wake();
        assert_eq!(woken.len(), 1, "one image freed, one present woken");
        assert_eq!(woken[0].0, request(2), "the one that asked first");
        assert_eq!(s.parked(), 1);
        s.ready(&tickets[1]).expect("drawn");
        s.queue(&tickets[1]).expect("head of the order now");
        s.complete(&tickets[1]).expect("shown");
        assert_eq!(s.wake()[0].0, request(3));
        assert_eq!(s.parked(), 0);
    }

    /// A request parks behind anything already parked even when an image is
    /// free, or a queue drains its way into presenting frames out of order.
    #[test]
    fn a_present_never_overtakes_one_already_parked() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(1, 1);
        let Admission::Acquired { ticket, .. } = s.submit(request(0)) else {
            panic!("the first image was free");
        };
        assert_eq!(s.submit(request(1)), Admission::Parked { ahead: 0 });
        s.ready(&ticket).expect("drawn");
        s.queue(&ticket).expect("head");
        s.complete(&ticket).expect("shown");
        // An image is free now, and the newcomer still goes behind.
        assert_eq!(s.submit(request(2)), Admission::Parked { ahead: 1 });
        let woken = s.wake();
        assert_eq!(woken.len(), 1);
        assert_eq!(woken[0].0, request(1), "the older request took the image");
    }

    /// The structural half of "backpressure does not head-of-line block":
    /// there is no queue for a slow display to fill that another surface
    /// reaches.
    #[test]
    fn a_full_surface_does_not_hold_another_ones_presents() {
        let mut slow = PresentStream::new(Order::Fifo);
        slow.configure(1, 1);
        let mut fast = PresentStream::new(Order::Fifo);
        fast.configure(2, 2);
        let Admission::Acquired { .. } = slow.submit(request(0)) else {
            panic!("the first image was free");
        };
        assert_eq!(slow.submit(request(1)), Admission::Parked { ahead: 0 });
        assert!(matches!(
            fast.submit(request(2)),
            Admission::Acquired { .. }
        ));
        assert_eq!(fast.parked(), 0);
    }

    /// Resizing while presents are parked: they belong to the surface, not to
    /// the swapchain, so a larger replacement is what lets them through.
    #[test]
    fn parked_presents_survive_a_swapchain_replacement() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(1, 1);
        let Admission::Acquired { ticket, .. } = s.submit(request(0)) else {
            panic!("the first image was free");
        };
        assert_eq!(s.submit(request(1)), Admission::Parked { ahead: 0 });
        let retired = s.replace(3, 3, at(7));
        assert_eq!(retired.generation, SwapchainGeneration::FIRST);
        assert_eq!(
            s.phase(&ticket),
            None,
            "the in-flight frame belonged to a swapchain that is gone"
        );
        let woken = s.wake();
        assert_eq!(woken.len(), 1, "the parked present takes a new image");
        assert_eq!(woken[0].0, request(1));
        assert_eq!(woken[0].1.generation, SwapchainGeneration::FIRST.next());
    }

    #[test]
    fn a_present_with_no_swapchain_is_not_parked() {
        let mut s = PresentStream::new(Order::Fifo);
        assert_eq!(s.submit(request(0)), Admission::NotConfigured);
        assert_eq!(
            s.parked(),
            0,
            "parking would be waiting for a swapchain that does not exist"
        );
    }

    #[test]
    fn abandoning_parked_presents_hands_them_back() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(1, 1);
        let Admission::Acquired { .. } = s.submit(request(0)) else {
            panic!("the first image was free");
        };
        assert_eq!(s.submit(request(1)), Admission::Parked { ahead: 0 });
        assert_eq!(s.submit(request(2)), Admission::Parked { ahead: 1 });
        assert_eq!(s.abandon_parked(), vec![request(1), request(2)]);
        assert_eq!(s.parked(), 0);
    }

    /// The backpressure census counts frames that waited, not polls that found
    /// nothing.
    #[test]
    fn waking_an_empty_queue_is_not_backpressure() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(1, 1);
        let Admission::Acquired { ticket, .. } = s.submit(request(0)) else {
            panic!("the first image was free");
        };
        assert_eq!(s.submit(request(1)), Admission::Parked { ahead: 0 });
        assert_eq!(s.census().2, 1, "one frame waited");
        for _ in 0..10 {
            assert!(s.wake().is_empty());
        }
        assert_eq!(s.census().2, 1, "and ten polls did not make it ten");
        s.ready(&ticket).expect("drawn");
        s.queue(&ticket).expect("head");
        s.complete(&ticket).expect("shown");
        assert_eq!(s.wake().len(), 1);
        assert_eq!(s.census().2, 1);
    }
}
