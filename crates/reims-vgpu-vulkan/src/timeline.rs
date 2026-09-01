//! The submission timeline: which value a submission signals, and what having
//! reached a value is allowed to mean.
//!
//! # Why the bookkeeping is separate from the semaphore
//!
//! A timeline semaphore is a monotone 64-bit counter the host advances, and
//! everything this rail defers — a command buffer's recycle, a native object's
//! destruction, an image's release — is deferred against a value on it. The
//! Vulkan part of that is two calls. The part that can be *wrong* is all
//! bookkeeping: handing out a value twice, reading a counter that went
//! backwards and believing it, or treating "I asked for this value" as "the
//! host reached it".
//!
//! So [`TimelineCursor`] is the bookkeeping with no `VkSemaphore` in it, and it
//! is exhaustively testable on a machine with no GPU. [`Timeline`] is the thin
//! wrapper that owns the handle and asks the driver.
//!
//! # Zero is not a point this rail hands out
//!
//! A timeline semaphore created with an initial value of zero has already
//! "reached" zero before anything is submitted. A retirement queued at zero
//! would therefore be collected on the first poll, before the work it was
//! waiting for was even recorded. [`TimelineCursor::reserve`] starts at one for
//! that reason, and it is why `TimelinePoint::default()` — which is zero — is a
//! value nothing here produces.
//!
//! # A counter that went backwards is refused, never clamped
//!
//! `vkGetSemaphoreCounterValue` on a healthy timeline is monotone. A value
//! below what was already observed means the handle is not the timeline the
//! caller thinks it is, or the device is in a state this rail cannot reason
//! about. Taking the larger of the two would hide it and keep retiring objects
//! against a fiction; [`Backwards`] names it, and the caller's answer is device
//! loss rather than a smaller number.
//!
//! # Reserving is not signalling
//!
//! [`TimelineCursor::reserve`] produces the value a submission *will* signal.
//! Nothing is reached until [`TimelineCursor::observe`] says the host got
//! there. The gap between them is exactly `outstanding`, and it is the number
//! that says how much work this rail is waiting on — not how much it queued.

use ash::vk;
use reims_vgpu_core::identity::TimelinePoint;

/// A timeline counter that went backwards.
///
/// Not recoverable by taking the larger value: the two readings cannot both
/// have come from a healthy timeline, so continuing would mean retiring
/// objects against a counter this rail has already been lied to about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Backwards {
    pub observed: u64,
    pub previously: u64,
}

impl Backwards {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        "vk_timeline_counter_went_backwards"
    }
}

impl std::fmt::Display for Backwards {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} observed={} previously={}",
            self.slug(),
            self.observed,
            self.previously
        )
    }
}

/// Which timeline values have been handed out, and which the host has reached.
///
/// No `VkSemaphore`: this is the part that can be wrong, and it is testable
/// without a device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimelineCursor {
    reserved: u64,
    reached: u64,
}

impl TimelineCursor {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            reserved: 0,
            reached: 0,
        }
    }

    /// The value the next submission will signal.
    ///
    /// Starts at one, never repeats, and says nothing about whether the host
    /// got there — see the module docs for why zero is not handed out.
    pub fn reserve(&mut self) -> TimelinePoint {
        self.reserved += 1;
        TimelinePoint(self.reserved)
    }

    /// The highest value handed out.
    #[must_use]
    pub const fn reserved(&self) -> TimelinePoint {
        TimelinePoint(self.reserved)
    }

    /// The highest value the host is known to have reached.
    #[must_use]
    pub const fn reached(&self) -> TimelinePoint {
        TimelinePoint(self.reached)
    }

    /// Record a counter reading from the driver.
    ///
    /// Returns the point now reached. A reading that repeats the previous one is
    /// fine and common — it means nothing finished since the last poll.
    ///
    /// # Errors
    ///
    /// If the reading is below what was already observed.
    pub fn observe(&mut self, counter: u64) -> Result<TimelinePoint, Backwards> {
        if counter < self.reached {
            return Err(Backwards {
                observed: counter,
                previously: self.reached,
            });
        }
        self.reached = counter;
        Ok(TimelinePoint(counter))
    }

    /// Whether the host has reached a point.
    ///
    /// False for a point that was never reserved, which is the safe answer: a
    /// caller asking about a value nothing will signal must not be told the
    /// work is done.
    #[must_use]
    pub const fn has_reached(&self, point: TimelinePoint) -> bool {
        self.reached >= point.0
    }

    /// Reserved values the host has not reached.
    ///
    /// What this rail is waiting on, not what it queued. A cursor whose
    /// `observe` has caught up to `reserve` has nothing outstanding even if a
    /// hundred submissions went through it.
    #[must_use]
    pub const fn outstanding(&self) -> u64 {
        self.reserved.saturating_sub(self.reached)
    }
}

/// One submission timeline: the semaphore and its bookkeeping.
///
/// Owns the handle. Destroying it is the caller's — this crate allocates no
/// Vulkan object it does not also hand back — so [`Timeline::semaphore`] is
/// what a destroy call takes and there is no `Drop` pretending to have a device
/// to call it with.
#[derive(Debug)]
pub struct Timeline {
    semaphore: vk::Semaphore,
    cursor: TimelineCursor,
}

impl Timeline {
    /// Adopt a timeline semaphore created with an initial value of zero.
    #[must_use]
    pub const fn adopt(semaphore: vk::Semaphore) -> Self {
        Self {
            semaphore,
            cursor: TimelineCursor::new(),
        }
    }

    #[must_use]
    pub const fn semaphore(&self) -> vk::Semaphore {
        self.semaphore
    }

    #[must_use]
    pub const fn cursor(&self) -> &TimelineCursor {
        &self.cursor
    }

    /// The value the next submission signals.
    pub fn reserve(&mut self) -> TimelinePoint {
        self.cursor.reserve()
    }

    #[must_use]
    pub const fn has_reached(&self, point: TimelinePoint) -> bool {
        self.cursor.has_reached(point)
    }

    #[must_use]
    pub const fn outstanding(&self) -> u64 {
        self.cursor.outstanding()
    }

    /// Ask the driver where the timeline is, and record it.
    ///
    /// # Errors
    ///
    /// The driver's own error, or [`Backwards`] when the reading is below one
    /// already taken.
    ///
    /// # Safety
    ///
    /// `device` must be the device the semaphore was created on, and the
    /// semaphore must not have been destroyed.
    pub unsafe fn poll(&mut self, device: &ash::Device) -> Result<TimelinePoint, PollFailure> {
        let counter = unsafe { device.get_semaphore_counter_value(self.semaphore) }
            .map_err(PollFailure::Driver)?;
        self.cursor.observe(counter).map_err(PollFailure::Backwards)
    }

    /// Wait for the host to reach a point, and record where it got to.
    ///
    /// A point the cursor already reached returns at once without a call: the
    /// driver cannot un-reach a value, so asking again is work with no answer
    /// in it.
    ///
    /// # Errors
    ///
    /// The driver's error — including `TIMEOUT`, which is not a failure of this
    /// rail and is passed through so the caller can decide — or [`Backwards`].
    ///
    /// # Safety
    ///
    /// As [`Timeline::poll`].
    pub unsafe fn wait(
        &mut self,
        device: &ash::Device,
        point: TimelinePoint,
        timeout_ns: u64,
    ) -> Result<(), PollFailure> {
        if self.cursor.has_reached(point) {
            return Ok(());
        }
        let semaphores = [self.semaphore];
        let values = [point.0];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe { device.wait_semaphores(&info, timeout_ns) }.map_err(PollFailure::Driver)?;
        // The wait succeeded, so the host is at least here. Recording it means
        // the next `has_reached` for this point answers without a call.
        self.cursor
            .observe(point.0)
            .map_err(PollFailure::Backwards)?;
        Ok(())
    }
}

/// Why a timeline reading did not produce a point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PollFailure {
    /// The driver refused. `TIMEOUT` arrives here too and is the caller's to
    /// interpret: a wait that timed out is not a device that failed.
    Driver(vk::Result),
    Backwards(Backwards),
}

impl PollFailure {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Driver(_) => "vk_timeline_driver_error",
            Self::Backwards(b) => b.slug(),
        }
    }
}

impl std::fmt::Display for PollFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Driver(r) => write!(f, "{} result={r:?}", self.slug()),
            Self::Backwards(b) => b.fmt(f),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A retirement queued at `TimelinePoint::default()` would be collected on
    /// the first poll of a fresh timeline, before the work it waits for was
    /// recorded. So zero is not a value this rail hands out.
    #[test]
    fn the_first_reserved_point_is_not_the_default_one() {
        let mut c = TimelineCursor::new();
        assert_eq!(c.reached(), TimelinePoint(0));
        assert!(
            c.has_reached(TimelinePoint::default()),
            "a fresh timeline has already reached zero"
        );
        let first = c.reserve();
        assert_eq!(first, TimelinePoint(1));
        assert!(!c.has_reached(first));
    }

    #[test]
    fn reserved_points_never_repeat() {
        let mut c = TimelineCursor::new();
        let points: Vec<TimelinePoint> = (0..64).map(|_| c.reserve()).collect();
        let mut sorted = points.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), points.len());
        assert_eq!(c.reserved(), TimelinePoint(64));
    }

    /// Reserving is not signalling, and `outstanding` says what is waited on
    /// rather than what was queued.
    #[test]
    fn outstanding_is_the_gap_between_reserved_and_reached() {
        let mut c = TimelineCursor::new();
        for _ in 0..5 {
            c.reserve();
        }
        assert_eq!(c.outstanding(), 5);
        assert_eq!(c.observe(3), Ok(TimelinePoint(3)));
        assert_eq!(c.outstanding(), 2);
        assert_eq!(c.observe(5), Ok(TimelinePoint(5)));
        assert_eq!(c.outstanding(), 0, "nothing is waited on");
        for _ in 0..100 {
            c.reserve();
        }
        c.observe(105).expect("the host caught up");
        assert_eq!(
            c.outstanding(),
            0,
            "a hundred submissions through it and nothing outstanding"
        );
    }

    /// A repeated reading is the ordinary case: nothing finished since the
    /// last poll.
    #[test]
    fn a_repeated_reading_is_not_an_error() {
        let mut c = TimelineCursor::new();
        c.reserve();
        c.reserve();
        assert_eq!(c.observe(1), Ok(TimelinePoint(1)));
        assert_eq!(c.observe(1), Ok(TimelinePoint(1)));
        assert_eq!(c.reached(), TimelinePoint(1));
    }

    /// Taking the larger of two readings would hide the fact that they cannot
    /// both have come from a healthy timeline, and keep retiring objects
    /// against a counter this rail has already been lied to about.
    #[test]
    fn a_counter_that_went_backwards_is_refused_and_not_clamped() {
        let mut c = TimelineCursor::new();
        for _ in 0..10 {
            c.reserve();
        }
        c.observe(7).expect("forwards");
        assert_eq!(
            c.observe(4),
            Err(Backwards {
                observed: 4,
                previously: 7
            })
        );
        assert_eq!(
            c.reached(),
            TimelinePoint(7),
            "and the refusal did not move the cursor either way"
        );
        assert_eq!(c.outstanding(), 3);
    }

    /// A caller asking about a value nothing will signal must not be told the
    /// work is done.
    #[test]
    fn an_unreserved_point_has_not_been_reached() {
        let mut c = TimelineCursor::new();
        c.reserve();
        c.observe(1).expect("the one submission finished");
        assert!(c.has_reached(TimelinePoint(1)));
        assert!(!c.has_reached(TimelinePoint(2)));
        assert!(!c.has_reached(TimelinePoint(u64::MAX)));
    }

    /// The point of the pairing: what `core::retire` defers against is what
    /// this hands out, and it collects exactly when the host gets there.
    #[test]
    fn a_retirement_deferred_against_a_reserved_point_collects_when_it_is_reached() {
        use reims_vgpu_core::identity::{DeviceEpoch, SessionGeneration};
        use reims_vgpu_core::retire::{Lifetime, NativeRetirement};

        let epoch = DeviceEpoch::FIRST;
        let lifetime = Lifetime::new(SessionGeneration::FIRST, epoch);
        let mut cursor = TimelineCursor::new();
        let mut queue: NativeRetirement<&str> = NativeRetirement::new();

        let first = cursor.reserve();
        let second = cursor.reserve();
        queue.queue(lifetime, first, "recorded first");
        queue.queue(lifetime, second, "recorded second");
        assert_eq!(
            queue.reached(epoch, cursor.reached()).len(),
            0,
            "nothing has been signalled, and zero must not collect either"
        );
        cursor.observe(1).expect("the first submission finished");
        let collected = queue.reached(epoch, cursor.reached());
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0].object, "recorded first");
        assert_eq!(queue.outstanding(), 1);
        cursor.observe(2).expect("and the second");
        assert_eq!(queue.reached(epoch, cursor.reached()).len(), 1);
        assert_eq!(queue.outstanding(), 0);
    }

    #[test]
    fn a_timeline_carries_its_handle_and_its_cursor() {
        let mut t = Timeline::adopt(vk::Semaphore::null());
        assert_eq!(t.semaphore(), vk::Semaphore::null());
        assert_eq!(t.reserve(), TimelinePoint(1));
        assert_eq!(t.outstanding(), 1);
        assert!(!t.has_reached(TimelinePoint(1)));
        assert_eq!(t.cursor().reserved(), TimelinePoint(1));
    }

    #[test]
    fn a_backwards_reading_names_itself_distinctly() {
        let b = Backwards {
            observed: 1,
            previously: 2,
        };
        assert_ne!(
            PollFailure::Driver(vk::Result::TIMEOUT).slug(),
            PollFailure::Backwards(b).slug()
        );
        assert!(PollFailure::Backwards(b)
            .to_string()
            .contains("previously=2"));
    }
}
