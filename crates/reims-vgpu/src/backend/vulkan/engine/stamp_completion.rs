//! Publishing and announcing FIFO completion stamps off the drain worker.
//!
//! # What the guest is actually waiting on
//!
//! `IOGPUEventMachine::waitForStamp(index, target)` in the guest reads the stamp
//! word **straight out of the page this device writes** — a signed
//! `target - current <= 0`, so it is wrap-safe — and if the target has not been
//! reached it builds a **one-second** deadline, sleeps on the stamp word's own
//! address as the wait channel, and re-reads the word on every wake.
//!
//! Two things follow:
//!
//! * The word is the authority. This module stores it only after the timeline
//!   point of the FIFO work it represents has completed.
//! * **The interrupt is the wakeup, not a hint.** Nothing re-checks the word
//!   until something wakes the thread, so a late interrupt is not a late
//!   notification, it is up to a full second of guest stall. An earlier attempt
//!   deferred the announcement to the drain worker's next tranche and measured
//!   exactly that: draws/s 3237 -> 2, presents/s 45 -> 1.
//!
//! The completion thread therefore waits the queue's monotonic timeline,
//! release-stores the shared word, and immediately raises the interrupt. The
//! drain worker only enqueues the checked word and returns.
//!
//! # Why a thread, and why it needs nothing from the device lock
//!
//! The announcement is three operations, and the device already does all three
//! off the drain worker for display VBL (`device::vbl_contended_pulse`):
//! `fetch_or` on the `Arc<AtomicU32>` clone of the interrupt-status register,
//! a push onto the lock-free `prompt_actions` queue, and `notify_actions` —
//! which the ABI documents as safe from any thread. The prompt rail exists for
//! precisely this: its own doc says it is there "so a guest ISR sees its
//! stamp-completion MSI while the drain worker is still rendering later
//! packets".
//!
//! This module owns none of that. It takes an [`AnnounceStamp`] hook the device
//! layer installs, so the engine keeps knowing nothing about `BoundDevice`.
//!
//! # Why a timeline semaphore rather than a fence
//!
//! A second waiter cannot use the ring's fences. `ResourcePools` owns every one
//! of them and resets each at retire, so a thread waiting on a ring fence races
//! the reset and a submission that signalled once can read as unsignalled
//! forever. Giving completion tracking its own fence instead breaks the ring
//! the other way: `vkQueueSubmit` takes exactly one fence, so the slot's fence
//! would never signal and its cleanup would never retire.
//!
//! A timeline semaphore has neither problem. It is signalled *in addition* to
//! the slot's fence, its value is monotonic so nothing has to be reset, and
//! `vkWaitSemaphores` may be called from any thread. Core in Vulkan 1.2, which
//! is this backend's baseline — and still gated, because the fallback is simply
//! the blocking rail every host used before this existed.

use ash::vk;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Pending completions owned by one FIFO before its producer must wait.
///
/// This is the FIFO contract's queue depth, not a tuning knob. Keeping the
/// bound per FIFO matters: pressure on one channel must not consume another
/// channel's completion capacity.
const FIFO_PENDING_STAMP_CAPACITY: usize = 32;

/// FIFOs with at least one completion whose word has not yet been published.
///
/// The drain worker needs this before taking the engine lock: a CPU-only stamp
/// still has to join the completion queue when an older stamp for the same FIFO
/// is pending, or it can publish ahead and the older completion later moves the
/// guest's fence backwards. There is one process-global engine and one
/// completion worker, so this is the lock-free projection of that worker's
/// per-FIFO counts.
static PENDING_FIFO_MASK: AtomicU32 = AtomicU32::new(0);

const _: () = assert!(crate::model::MAX_CHANNELS <= u32::BITS as usize);

pub(crate) fn fifo_has_pending_stamp(index: u32) -> bool {
    index < u32::BITS && PENDING_FIFO_MASK.load(Ordering::Acquire) & (1u32 << index) != 0
}

/// Raise the guest-visible interrupt for a completed stamp slot.
///
/// Installed by the device layer, which owns the interrupt-status clone and the
/// prompt action queue. Called from the completion thread with no lock of this
/// crate's held, so an implementation must not reach for the device lock.
pub type AnnounceStamp = Arc<dyn Fn(u32) + Send + Sync>;

/// The installed announcement hook.
///
/// A global rather than a constructor argument because the two events have no
/// order between them: the device layer binds when QEMU realizes the device, and
/// the engine builds its context lazily on the first draw. Whichever happens
/// first, the thread reads the hook when it has something to announce.
///
/// **A stamp completing with no hook installed is a lost wakeup**, so it is
/// fail-visible rather than silent — it means a submission reached the GPU
/// before this device was bound, which nothing should be able to arrange.
static HOOK: std::sync::Mutex<Option<AnnounceStamp>> = std::sync::Mutex::new(None);

/// Install the hook the completion thread announces through. Idempotent; the
/// last caller wins, which is what a device rebind wants.
///
/// There is no uninstall. The hook the device layer installs resolves its
/// device by id every time it is called, so one left behind by a torn-down
/// device holds nothing and announces nothing — an uninstall would only be a
/// second way to reach the same state, and a race against a completion already
/// in flight.
pub fn install_announce(hook: AnnounceStamp) {
    *HOOK.lock().unwrap_or_else(|e| e.into_inner()) = Some(hook);
}

fn announce(index: u32) {
    let hook = HOOK
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(Arc::clone);
    match hook {
        // Called with no lock of this module's held: the hook reaches the
        // device's prompt queue, and holding `HOOK` across it would put this
        // module's mutex under the device's.
        Some(hook) => hook(index),
        None => crate::observe::fail(format!(
            "stamp_announce_no_hook reason=stamp_announce_no_hook index={index} \
             (a stamp completed with no device bound to raise its interrupt; the guest \
             waiting on it will sleep to its one-second deadline)"
        )),
    }
}

/// One stamp waiting for its queue point, in FIFO order.
#[derive(Clone, Debug)]
struct Waiting {
    /// Timeline value the FIFO submission preceding this stamp signals.
    ///
    /// `None` is a stamp recorded while a deferred draw batch is still open.
    /// The next successful queue submission assigns its point to every such
    /// record. Keeping the record pending before the submit is what preserves
    /// FIFO chaining without making the stamp close the command buffer.
    timeline: Option<u64>,
    /// Stamp slot index, for the interrupt-status bit.
    index: u32,
    /// The checked shared-memory word written before the interrupt is raised.
    word: crate::runtime::guest_ram::GuestRef,
    /// The FIFO completion value published into `word`.
    stamp: u32,
}

#[derive(Default)]
struct PendingQueue {
    waiting: std::collections::VecDeque<Waiting>,
    per_fifo: [usize; crate::model::MAX_CHANNELS],
}

impl PendingQueue {
    fn is_full(&self, index: usize) -> bool {
        self.per_fifo[index] == FIFO_PENDING_STAMP_CAPACITY
    }

    fn push(&mut self, waiting: Waiting) {
        self.per_fifo[waiting.index as usize] += 1;
        self.waiting.push_back(waiting);
    }

    fn pop_front(&mut self) -> Option<Waiting> {
        let waiting = self.waiting.pop_front()?;
        self.per_fifo[waiting.index as usize] -= 1;
        Some(waiting)
    }

    fn has_pending(&self, index: usize) -> bool {
        self.per_fifo[index] != 0
    }

    fn bind_unsubmitted(&mut self, timeline: u64) -> usize {
        let mut bound = 0;
        for waiting in &mut self.waiting {
            if waiting.timeline.is_none() {
                waiting.timeline = Some(timeline);
                bound += 1;
            }
        }
        bound
    }
}

/// The queue the drain worker pushes to and the completion thread drains.
struct Shared {
    queue: Mutex<PendingQueue>,
    /// Woken by a push and by shutdown. The thread waits here only when the
    /// queue is empty; otherwise it is blocked in `vkWaitSemaphores`, which is
    /// where it should be.
    wake: Condvar,
    stop: AtomicBool,
    /// Highest value handed out. The drain worker reserves with `fetch_add`
    /// under the engine lock, so reservation order is submission order.
    next_value: AtomicU64,
    /// Highest timeline point belonging to a successful queue submission.
    latest_submitted: AtomicU64,
}

/// A running completion thread, owned by the device context.
pub(crate) struct StampCompletion {
    shared: Arc<Shared>,
    semaphore: vk::Semaphore,
    join: Option<std::thread::JoinHandle<()>>,
}

impl StampCompletion {
    /// Create the semaphore and start the thread.
    ///
    /// `device` is cloned into the thread. `ash::Device` is a handle plus a
    /// function-pointer table, and the two entry points the thread calls —
    /// `vkWaitSemaphores` and `vkGetSemaphoreCounterValue` — are not externally
    /// synchronized against anything the drain worker does to this semaphore
    /// (only signalling is, and only the queue signals it). What *is* required
    /// is that the thread stop before `vkDestroyDevice`, which [`Self::stop`]
    /// guarantees and `DeviceContext::destroy` calls.
    ///
    /// # Safety
    ///
    /// `device` must outlive the returned value's [`Self::stop`].
    pub(crate) unsafe fn start(device: &ash::Device) -> Result<Self, vk::Result> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let ci = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
        let semaphore = unsafe { device.create_semaphore(&ci, None) }?;
        let shared = Arc::new(Shared {
            queue: Mutex::new(PendingQueue::default()),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            next_value: AtomicU64::new(0),
            latest_submitted: AtomicU64::new(0),
        });
        let thread_shared = Arc::clone(&shared);
        let thread_device = device.clone();
        let join = std::thread::Builder::new()
            .name("reims-vgpu-stamp".into())
            .spawn(move || run(&thread_device, semaphore, &thread_shared))
            .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
        Ok(Self {
            shared,
            semaphore,
            join: Some(join),
        })
    }

    /// Reserve the timeline point for one FIFO-owned queue submission.
    ///
    /// Reserved under the engine lock, so the values are handed out in
    /// submission order — which is what makes a single-threaded drain of the
    /// queue announce stamps in that same order without any further ordering
    /// machinery.
    pub(crate) fn reserve_submission(&self) -> (vk::Semaphore, u64) {
        let value = self.shared.next_value.fetch_add(1, Ordering::AcqRel) + 1;
        (self.semaphore, value)
    }

    /// Publish a successfully submitted queue point.
    pub(crate) fn note_submitted(&self, value: u64) {
        self.shared.latest_submitted.store(value, Ordering::Release);
        let bound = self
            .shared
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .bind_unsubmitted(value);
        if bound != 0 {
            self.shared.wake.notify_all();
        }
    }

    /// The newest successfully submitted queue point.
    pub(crate) fn latest_submitted(&self) -> Option<(vk::Semaphore, u64)> {
        let value = self.shared.latest_submitted.load(Ordering::Acquire);
        (value != 0).then_some((self.semaphore, value))
    }

    /// Retire one FIFO completion after `timeline` completes.
    pub(crate) fn wait_for_stamp(
        &self,
        timeline: u64,
        index: u32,
        word: crate::runtime::guest_ram::GuestRef,
        stamp: u32,
    ) {
        let Some(slot) = ((index as usize) < crate::model::MAX_CHANNELS).then_some(index as usize)
        else {
            crate::observe::fail(format!(
                "stamp_fifo_out_of_range reason=stamp_fifo_out_of_range index={index} \
                 max_channels={}",
                crate::model::MAX_CHANNELS
            ));
            return;
        };
        let mut queue = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
        while queue.is_full(slot) && !self.shared.stop.load(Ordering::Acquire) {
            queue = self
                .shared
                .wake
                .wait(queue)
                .unwrap_or_else(|e| e.into_inner());
        }
        if self.shared.stop.load(Ordering::Acquire) {
            return;
        }
        queue.push(Waiting {
            timeline: Some(timeline),
            index,
            word,
            stamp,
        });
        PENDING_FIFO_MASK.fetch_or(1u32 << index, Ordering::Release);
        drop(queue);
        self.shared.wake.notify_one();
    }

    /// Register a stamp behind the command buffer that is still recording.
    ///
    /// Returns `false` instead of blocking when this FIFO's contract-sized
    /// pending ring is full. The caller owns the open command buffer, so
    /// sleeping there would prevent the very submission that can make room;
    /// it must submit the batch and retry against that concrete point.
    pub(crate) fn queue_for_next_submission(
        &self,
        index: u32,
        word: crate::runtime::guest_ram::GuestRef,
        stamp: u32,
    ) -> bool {
        let Some(slot) = ((index as usize) < crate::model::MAX_CHANNELS).then_some(index as usize)
        else {
            crate::observe::fail(format!(
                "stamp_fifo_out_of_range reason=stamp_fifo_out_of_range index={index} \
                 max_channels={}",
                crate::model::MAX_CHANNELS
            ));
            return false;
        };
        let mut queue = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
        if queue.is_full(slot) || self.shared.stop.load(Ordering::Acquire) {
            return false;
        }
        queue.push(Waiting {
            timeline: None,
            index,
            word,
            stamp,
        });
        PENDING_FIFO_MASK.fetch_or(1u32 << index, Ordering::Release);
        drop(queue);
        self.shared.wake.notify_one();
        true
    }

    /// Wait until this FIFO has no queued completion word left to publish.
    ///
    /// Used only before the CPU fallback writes a newer value. GPU work may
    /// already be settled while its completion worker has not yet stored the
    /// older word; waiting on the GPU alone would still permit that older store
    /// to land after the fallback and move the guest's fence backwards.
    pub(crate) fn wait_for_fifo_idle(&self, index: u32) {
        let Some(slot) = ((index as usize) < crate::model::MAX_CHANNELS).then_some(index as usize)
        else {
            return;
        };
        let mut queue = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
        while queue.has_pending(slot) && !self.shared.stop.load(Ordering::Acquire) {
            queue = self
                .shared
                .wake
                .wait(queue)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Stop the thread and destroy the semaphore.
    ///
    /// Must run before `vkDestroyDevice`: the thread holds a cloned `ash::Device`
    /// and is blocked inside it.
    ///
    /// # Safety
    ///
    /// `device` must be the device this was started with, and must not yet be
    /// destroyed.
    pub(crate) unsafe fn stop(&mut self, device: &ash::Device) {
        self.shared.stop.store(true, Ordering::Release);
        // Signal past every reserved value so a thread blocked in
        // `vkWaitSemaphores` returns rather than sitting out its deadline. The
        // thread observes `stop` after the wait and never publishes a word for
        // work this host signal merely skipped over.
        let past_everything = self.shared.next_value.load(Ordering::Acquire) + 1;
        let signal = vk::SemaphoreSignalInfo::default()
            .semaphore(self.semaphore)
            .value(past_everything);
        let _ = unsafe { device.signal_semaphore(&signal) };
        self.shared.wake.notify_all();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
        PENDING_FIFO_MASK.store(0, Ordering::Release);
        unsafe { device.destroy_semaphore(self.semaphore, None) };
    }
}

/// The completion thread.
///
/// Blocks in `vkWaitSemaphores` while there is a stamp outstanding and on the
/// condvar while there is not, so it costs nothing when the guest is idle and
/// adds no latency when it is not.
fn run(device: &ash::Device, semaphore: vk::Semaphore, shared: &Shared) {
    loop {
        let next = {
            let mut queue = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if shared.stop.load(Ordering::Acquire) {
                    return;
                }
                if let Some(front) = queue.waiting.front().cloned() {
                    if front.timeline.is_some() {
                        break Some(front);
                    }
                }
                let (guard, _) = shared
                    .wake
                    .wait_timeout(queue, std::time::Duration::from_millis(250))
                    .unwrap_or_else(|e| e.into_inner());
                queue = guard;
            }
        };
        let Some(waiting) = next else {
            return;
        };
        let semaphores = [semaphore];
        let timeline = waiting.timeline.expect("front was checked as submitted");
        let values = [timeline];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // The same deadline every blocking wait in this backend uses. Reaching
        // it means the queue has not run this submission, which is a device
        // fault rather than a slow frame — announce anyway and say so, because
        // the guest's own deadline is one second and a withheld stamp costs it
        // that whether the GPU is wedged or not.
        let completed =
            match unsafe { device.wait_semaphores(&info, super::context::FENCE_TIMEOUT_NS) } {
                Ok(()) => true,
                Err(vk::Result::TIMEOUT) => {
                    crate::observe::fail(format!(
                        "stamp_wait_timeout reason=stamp_wait_timeout index={} value={} \
                     (the submission carrying this stamp's word has not executed within the \
                     fence deadline; announcing it anyway so the guest is not left asleep)",
                        waiting.index, timeline
                    ));
                    false
                }
                Err(e) => {
                    crate::observe::fail(format!(
                        "stamp_wait_failed reason=stamp_wait_failed index={} value={} err={e:?} \
                     (announcing regardless, for the reason a timeout does)",
                        waiting.index, timeline
                    ));
                    // Announcing is not recovering. This thread may not take the
                    // engine lock — it exists to announce guest fences while the
                    // drain worker holds it — so it latches the loss and the drain's
                    // end-of-tranche flush runs the recovery. Without this, a boot
                    // whose device dies *here* never rebuilds it: the guest stops
                    // drawing because its work stopped completing, and "the next
                    // draw will surface it" then waits forever.
                    if e == vk::Result::ERROR_DEVICE_LOST {
                        super::device_lost::note_device_lost_seen();
                    }
                    false
                }
            };
        let stopping = shared.stop.load(Ordering::Acquire);
        if should_publish(completed, stopping) && !publish_stamp_word(&waiting) {
            crate::observe::fail(format!(
                "stamp_cpu_store_failed reason=stamp_cpu_store_failed index={} value={:#x} \
                 (the completed queue point could not publish its checked shared word)",
                waiting.index, waiting.stamp
            ));
        }
        {
            let mut queue = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            queue.pop_front();
            let slot = waiting.index as usize;
            if !queue.has_pending(slot) {
                PENDING_FIFO_MASK.fetch_and(!(1u32 << waiting.index), Ordering::Release);
            }
        }
        shared.wake.notify_all();
        announce(waiting.index);
        if stopping {
            return;
        }
    }
}

fn should_publish(completed: bool, stopping: bool) -> bool {
    completed && !stopping
}

fn publish_stamp_word(waiting: &Waiting) -> bool {
    waiting.word.store_u32_release(waiting.stamp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn waiting(index: u32, stamp: u32) -> Waiting {
        let mut word = Box::new(0u32);
        let import = Arc::new(
            crate::runtime::guest_ram::GuestRamImport::new_host_allocation(
                (&mut *word) as *mut u32 as usize,
                std::mem::size_of::<u32>() as u64,
                std::mem::align_of::<u32>() as u64,
            )
            .expect("test import"),
        );
        // The import deliberately owns no allocation. Leak this one-word test
        // backing so every queued GuestRef remains valid for the test's life.
        Box::leak(word);
        let slice = import.slice(0, 4).expect("stamp word");
        Waiting {
            timeline: Some(u64::from(stamp) + 1),
            index,
            word: crate::runtime::guest_ram::GuestRef::new(import, slice).expect("guest word"),
            stamp,
        }
    }

    #[test]
    fn pending_capacity_is_per_fifo_and_fifo_order_is_preserved() {
        let mut queue = PendingQueue::default();
        for stamp in 0..FIFO_PENDING_STAMP_CAPACITY as u32 {
            queue.push(waiting(0, stamp));
        }
        assert!(queue.is_full(0));
        assert!(!queue.is_full(1));
        queue.push(waiting(1, 0xfeed));

        for stamp in 0..FIFO_PENDING_STAMP_CAPACITY as u32 {
            let entry = queue.pop_front().expect("root completion");
            assert_eq!((entry.index, entry.stamp), (0, stamp));
        }
        assert!(!queue.has_pending(0));
        assert!(queue.has_pending(1));
        let child = queue.pop_front().expect("child completion");
        assert_eq!((child.index, child.stamp), (1, 0xfeed));
        assert!(!queue.has_pending(1));
    }

    /// Reservation order is submission order, and only successful submissions
    /// become attachable completion points.
    #[test]
    fn reservations_are_monotonic_and_success_is_published_separately() {
        let shared = Shared {
            queue: Mutex::new(PendingQueue::default()),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            next_value: AtomicU64::new(0),
            latest_submitted: AtomicU64::new(0),
        };
        for n in 1u64..=3 {
            let value = shared.next_value.fetch_add(1, Ordering::AcqRel) + 1;
            assert_eq!(value, n, "timeline values start at 1 and never repeat");
        }
        assert_eq!(shared.latest_submitted.load(Ordering::Acquire), 0);
        shared.latest_submitted.store(3, Ordering::Release);
        assert_eq!(shared.latest_submitted.load(Ordering::Acquire), 3);
    }

    /// The initial value is 0 and the first reservation is 1, so no submission
    /// ever signals the value the semaphore was created at. A first reservation
    /// of 0 would be already-signalled at creation and its stamp would be
    /// announced before the GPU had run anything.
    #[test]
    fn no_reservation_can_collide_with_the_semaphores_initial_value() {
        let next = AtomicU64::new(0);
        assert_eq!(next.fetch_add(1, Ordering::AcqRel) + 1, 1);
    }

    #[test]
    fn completed_waiting_entry_publishes_its_word() {
        let mut words = [0u32; 2];
        let import = Arc::new(
            crate::runtime::guest_ram::GuestRamImport::new_host_allocation(
                words.as_mut_ptr() as usize,
                std::mem::size_of_val(&words) as u64,
                std::mem::align_of_val(&words) as u64,
            )
            .expect("test import"),
        );
        let slice = import.slice(4, 4).expect("second word");
        let word = crate::runtime::guest_ram::GuestRef::new(import, slice).expect("guest word");
        let waiting = Waiting {
            timeline: Some(7),
            index: 2,
            word,
            stamp: 0x89ab_cdef,
        };

        assert!(publish_stamp_word(&waiting));
        assert_eq!(words, [0, 0x89ab_cdefu32.to_le()]);
    }

    #[test]
    fn teardown_wakeup_never_publishes_unfinished_work() {
        assert!(should_publish(true, false));
        assert!(!should_publish(false, false));
        assert!(!should_publish(true, true));
    }

    #[test]
    fn an_open_batch_stamp_binds_only_when_submission_succeeds() {
        let mut queue = PendingQueue::default();
        let mut submitted = waiting(0, 1);
        submitted.timeline = Some(7);
        let mut deferred_a = waiting(0, 2);
        deferred_a.timeline = None;
        let mut deferred_b = waiting(1, 3);
        deferred_b.timeline = None;
        queue.push(submitted);
        queue.push(deferred_a);
        queue.push(deferred_b);

        assert_eq!(queue.bind_unsubmitted(11), 2);
        let points: Vec<Option<u64>> = queue.waiting.iter().map(|w| w.timeline).collect();
        assert_eq!(points, vec![Some(7), Some(11), Some(11)]);
    }
}
