//! One native recording, and the single value that carries everything it owns
//! from preparation to cleanup.
//!
//! # Why this is one value and not seven
//!
//! A recording needs its allocating worker, that worker's queue family, its
//! command pool, the command buffers it recorded into, the descriptor sets it
//! wrote, and the native pipelines it named. Every one of those has a lifetime
//! that ends at the same moment — when the timeline says the GPU is done with
//! this submission — and every one of them is a use-after-free if it ends
//! earlier.
//!
//! Held as separate values threaded through the same call chain, that is six
//! chances to drop one, and the shape of the failure is a pipeline destroyed
//! while a command buffer still names it: a crash a long way from its cause,
//! usually in the driver. So they are one value here, moved between the four
//! states below, and the compiler is what stops a stage from losing part of it.
//! Every refusal returns the whole value rather than a code, so a failed stage
//! leaves the caller holding exactly what it held before.
//!
//! # The four states
//!
//! [`Preparation`] takes the command-buffer slots and accumulates descriptor
//! emissions and pipeline variants. [`Recorded`] is a complete recording that
//! has not been handed to the driver. [`InFlight`] has a [`Receipt`] and is
//! waiting on the timeline. [`Retired`] is past its point and may be taken
//! apart. Each transition consumes the previous state, so none of them can be
//! observed twice or skipped.
//!
//! # Failure unwinds the prefix rather than leaking it
//!
//! [`Preparation::begin`] asking for several buffers and getting three of four
//! must not leave three slots leased to nobody: the ring would never hand them
//! out again and the worker would silently lose depth every time it was busy.
//! So a partial acquisition abandons what it took before returning, and a
//! preparation abandoned later hands back its slots *and* its descriptor
//! emissions through [`Unwound`].
//!
//! # A forged worker refuses before anything moves
//!
//! Every transition that touches worker-owned state resolves the recording's
//! own [`WorkerId`] against the live [`WorkerPools`] and checks that the
//! family and the pool handle are the ones the recording was built from. A
//! mismatch means the recording came from a different worker, a different
//! epoch's pools, or a value that was assembled rather than prepared — and in
//! all three cases the right answer is to touch no Vulkan handle and give the
//! caller its recording back untouched. The refusal carries the full value for
//! exactly that reason, and the tests assert that the pools compare equal
//! before and after.
//!
//! # No Vulkan call
//!
//! This module holds handles and makes none of the calls that consume them, so
//! the whole state machine above is tested on a machine with no GPU.

// Every refusal here returns the whole recording, which is the module's claim:
// a stage that will not take the work must not be able to keep a piece of it.
// That makes the `Err` variant as large as the recording by construction, and
// boxing it would put a heap allocation on the failure path in exchange for
// nothing — the value is moved, not copied, on both arms.
#![allow(clippy::result_large_err)]

use ash::vk;
use reims_vgpu_core::executor::WorkerId;
use reims_vgpu_core::identity::{DeviceEpoch, TimelinePoint};

use crate::descriptor::SetEmission;
use crate::pools::{Exhausted, Lease, WorkerPools};
use crate::submission::Receipt;
use crate::variant::Variant;

/// Why a recording was not admitted by the state it was offered to.
///
/// Always accompanied by the recording itself, and never by a partial effect:
/// each of these is decided before anything is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mismatch {
    /// The recording names a worker this population does not have.
    UnknownWorker { worker: WorkerId, population: usize },
    /// The named worker exists but submits to a different queue family than
    /// the recording was built for. A submission to the wrong family is
    /// invalid usage the driver is not required to diagnose.
    WrongFamily { expected: u32, found: u32 },
    /// The named worker's pool is not the pool this recording was allocated
    /// from — the worker's pools were rebuilt underneath it.
    WrongPool,
    /// The recording belongs to a device epoch this one is not.
    WrongEpoch {
        expected: DeviceEpoch,
        found: DeviceEpoch,
    },
}

impl Mismatch {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::UnknownWorker { .. } => "vk_recording_unknown_worker",
            Self::WrongFamily { .. } => "vk_recording_wrong_family",
            Self::WrongPool => "vk_recording_wrong_pool",
            Self::WrongEpoch { .. } => "vk_recording_wrong_epoch",
        }
    }
}

impl std::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownWorker { worker, population } => write!(
                f,
                "{} worker={} population={}",
                self.slug(),
                worker.0,
                population
            ),
            Self::WrongFamily { expected, found } => {
                write!(f, "{} expected={expected} found={found}", self.slug())
            }
            Self::WrongPool => write!(f, "{}", self.slug()),
            Self::WrongEpoch { expected, found } => write!(
                f,
                "{} expected={} found={}",
                self.slug(),
                expected.get(),
                found.get()
            ),
        }
    }
}

/// Why a preparation could not begin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Begin {
    /// The worker does not exist. Nothing was taken.
    Worker(Mismatch),
    /// The worker has no free command-buffer slot. Whatever prefix this
    /// preparation had already taken has been given back.
    Depth(Exhausted),
}

impl std::fmt::Display for Begin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Worker(mismatch) => mismatch.fmt(f),
            Self::Depth(exhausted) => exhausted.fmt(f),
        }
    }
}

/// What every state of a recording owns, whatever state it is in.
///
/// Private, and never destructured outside this module: the states below are
/// the only way to reach it, which is what makes "a stage cannot lose part of
/// the recording" a compiler fact rather than a review comment.
#[derive(Debug)]
struct Held<V> {
    epoch: DeviceEpoch,
    worker: WorkerId,
    family: u32,
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
    emissions: Vec<SetEmission>,
    variants: Vec<Variant<V>>,
}

impl<V> Held<V> {
    /// Check this recording against the live pools without touching either.
    fn resolve(&self, pools: &WorkerPools) -> Result<(), Mismatch> {
        let Some(pool) = pools.of_worker(self.worker) else {
            return Err(Mismatch::UnknownWorker {
                worker: self.worker,
                population: pools.population(),
            });
        };
        if pool.family() != self.family {
            return Err(Mismatch::WrongFamily {
                expected: self.family,
                found: pool.family(),
            });
        }
        if pool.pool() != self.pool {
            return Err(Mismatch::WrongPool);
        }
        Ok(())
    }
}

/// A recording being built: its slots are held, its descriptors and pipelines
/// accumulate.
#[derive(Debug)]
#[must_use = "a preparation that is neither finished nor unwound leaks its command-buffer slots"]
pub struct Preparation<V> {
    held: Held<V>,
    leases: Vec<Lease>,
}

impl<V> Preparation<V> {
    /// Take `buffers` command-buffer slots from `worker` and start a recording.
    ///
    /// # Errors
    ///
    /// [`Begin::Worker`] when the worker is not in this population, and
    /// [`Begin::Depth`] when it ran out of slots — in which case the slots
    /// already taken by this call have been given back, so the ring is exactly
    /// as it was.
    ///
    /// # Panics
    ///
    /// If `buffers` is zero. A recording with no command buffer records
    /// nothing and would still consume a timeline point.
    pub fn begin(
        epoch: DeviceEpoch,
        worker: WorkerId,
        pools: &mut WorkerPools,
        buffers: usize,
    ) -> Result<Self, Begin> {
        assert!(buffers > 0, "a recording with no command buffer is not one");
        let population = pools.population();
        let Some(pool) = pools.for_worker(worker) else {
            return Err(Begin::Worker(Mismatch::UnknownWorker {
                worker,
                population,
            }));
        };

        let mut leases = Vec::with_capacity(buffers);
        for _ in 0..buffers {
            match pool.ring_mut().begin() {
                Ok(lease) => leases.push(lease),
                Err(exhausted) => {
                    // The prefix goes back before the refusal leaves: a slot
                    // leased to a preparation that never existed is depth the
                    // worker never gets back.
                    for lease in leases.drain(..).rev() {
                        pool.ring_mut().abandon(lease);
                    }
                    return Err(Begin::Depth(exhausted));
                }
            }
        }

        let handles = leases.iter().map(|lease| pool.buffer(lease)).collect();
        Ok(Self {
            held: Held {
                epoch,
                worker,
                family: pool.family(),
                pool: pool.pool(),
                buffers: handles,
                emissions: Vec::new(),
                variants: Vec::new(),
            },
            leases,
        })
    }

    /// The command buffers to record into, in the order they were taken.
    #[must_use]
    pub fn buffers(&self) -> &[vk::CommandBuffer] {
        &self.held.buffers
    }

    #[must_use]
    pub const fn worker(&self) -> WorkerId {
        self.held.worker
    }

    #[must_use]
    pub const fn family(&self) -> u32 {
        self.held.family
    }

    /// A descriptor emission this recording planned. Held so that unwinding
    /// can tell the ring the write never happened.
    pub fn claimed(&mut self, emission: SetEmission) {
        self.held.emissions.push(emission);
    }

    /// A native pipeline this recording named. Held so that the pipeline
    /// outlives the submission that reads it, whatever the guest does to the
    /// semantic pipeline in the meantime.
    pub fn named(&mut self, variant: Variant<V>) {
        self.held.variants.push(variant);
    }

    /// Recording is finished; the work is ready for the queue.
    pub fn finish(self) -> Recorded<V> {
        Recorded {
            held: self.held,
            leases: self.leases,
        }
    }

    /// Give up before submitting. See [`Unwound`].
    pub fn unwind(self) -> Unwound<V> {
        Unwound {
            leases: self.leases,
            emissions: self.held.emissions,
            variants: self.held.variants,
        }
    }
}

/// A complete recording that the driver has not been offered yet.
#[derive(Debug)]
#[must_use = "a recorded submission that is neither accepted nor refused leaks its command-buffer slots"]
pub struct Recorded<V> {
    held: Held<V>,
    leases: Vec<Lease>,
}

impl<V> Recorded<V> {
    #[must_use]
    pub fn buffers(&self) -> &[vk::CommandBuffer] {
        &self.held.buffers
    }

    #[must_use]
    pub const fn worker(&self) -> WorkerId {
        self.held.worker
    }

    #[must_use]
    pub const fn family(&self) -> u32 {
        self.held.family
    }

    #[must_use]
    pub const fn epoch(&self) -> DeviceEpoch {
        self.held.epoch
    }

    /// The driver took the work.
    ///
    /// The slots become readable by the GPU until the receipt's point, which is
    /// what stops the next recording from being written into them.
    ///
    /// # Errors
    ///
    /// [`Mismatch`] with the recording returned untouched, and the pools
    /// unchanged.
    pub fn accepted(
        self,
        receipt: Receipt,
        pools: &mut WorkerPools,
    ) -> Result<InFlight<V>, (Self, Mismatch)> {
        if let Err(mismatch) = self.held.resolve(pools) {
            return Err((self, mismatch));
        }
        let pool = pools
            .for_worker(self.held.worker)
            .expect("resolve admitted this worker");
        for lease in self.leases {
            pool.ring_mut().submitted(lease, receipt.point());
        }
        Ok(InFlight {
            held: self.held,
            receipt,
        })
    }

    /// The driver refused the submission, so it never read the buffers.
    ///
    /// The slots go straight back to free rather than through the timeline: no
    /// point was signalled for work the queue never took, so waiting on one
    /// would strand the worker's whole depth.
    ///
    /// # Errors
    ///
    /// [`Mismatch`] with the recording returned untouched.
    pub fn refused(self, pools: &mut WorkerPools) -> Result<Unwound<V>, (Self, Mismatch)> {
        if let Err(mismatch) = self.held.resolve(pools) {
            return Err((self, mismatch));
        }
        let pool = pools
            .for_worker(self.held.worker)
            .expect("resolve admitted this worker");
        for lease in self.leases {
            pool.ring_mut().abandon(lease);
        }
        Ok(Unwound {
            leases: Vec::new(),
            emissions: self.held.emissions,
            variants: self.held.variants,
        })
    }
}

/// What a preparation or a refused submission gives back.
///
/// The leases are already back in their ring when this came from
/// [`Recorded::refused`]; they are still outstanding when it came from
/// [`Preparation::unwind`], which is why [`Unwound::restore`] takes the pools.
/// The descriptor emissions are always outstanding: nothing else knows the
/// write did not happen.
#[derive(Debug)]
#[must_use = "an unwound recording that is not restored leaks command-buffer slots and descriptor sets"]
pub struct Unwound<V> {
    leases: Vec<Lease>,
    emissions: Vec<SetEmission>,
    variants: Vec<Variant<V>>,
}

impl<V> Unwound<V> {
    /// The descriptor emissions whose writes never completed. Each must be
    /// handed to [`crate::descriptor::SetRing::abandoned`] on the ring that
    /// planned it; this value does not know which ring that is, and inventing
    /// one here would be the guess that binds a stale set.
    pub fn emissions(&self) -> &[SetEmission] {
        &self.emissions
    }

    /// The native pipelines this recording had named. Held rather than
    /// released early: the driver may have refused a submission whose buffers
    /// it had already begun reading, and a pipeline destroyed on that path is
    /// the use-after-free the whole module exists to prevent. They are released
    /// when this value is dropped.
    #[must_use]
    pub fn variants(&self) -> &[Variant<V>] {
        &self.variants
    }

    /// Return the command-buffer slots to their worker and drop the pipelines
    /// this recording was holding alive.
    ///
    /// Returns the emissions, because they belong to descriptor rings this
    /// value cannot name.
    ///
    /// # Errors
    ///
    /// [`Mismatch`] when the worker cannot be resolved, with the unwound
    /// recording returned intact. There is no partial restore: nothing is given
    /// back until the worker is proven.
    pub fn restore(
        mut self,
        worker: WorkerId,
        pools: &mut WorkerPools,
    ) -> Result<Vec<SetEmission>, (Self, Mismatch)> {
        let population = pools.population();
        let Some(pool) = pools.for_worker(worker) else {
            return Err((self, Mismatch::UnknownWorker { worker, population }));
        };
        for lease in self.leases.drain(..).rev() {
            pool.ring_mut().abandon(lease);
        }
        Ok(self.emissions)
    }
}

/// A recording the driver accepted, waiting for the timeline to pass it.
#[derive(Debug)]
#[must_use = "an in-flight recording that is dropped destroys pipelines the GPU is still reading"]
pub struct InFlight<V> {
    held: Held<V>,
    receipt: Receipt,
}

impl<V> InFlight<V> {
    #[must_use]
    pub const fn point(&self) -> TimelinePoint {
        self.receipt.point()
    }

    #[must_use]
    pub const fn worker(&self) -> WorkerId {
        self.held.worker
    }

    #[must_use]
    pub const fn epoch(&self) -> DeviceEpoch {
        self.held.epoch
    }
}

/// A recording whose timeline point the GPU has passed.
#[derive(Debug)]
#[must_use = "a retired recording that is dropped without release skips the live-pool check"]
pub struct Retired<V> {
    held: Held<V>,
    at: TimelinePoint,
}

impl<V> Retired<V> {
    /// The point the GPU reached that made this retirement true.
    #[must_use]
    pub const fn at(&self) -> TimelinePoint {
        self.at
    }

    #[must_use]
    pub const fn worker(&self) -> WorkerId {
        self.held.worker
    }

    /// The descriptor emissions this recording wrote. The rings free them
    /// against the same timeline reading; this is what they wrote, for a caller
    /// that reports rather than frees.
    pub fn emissions(&self) -> &[SetEmission] {
        &self.held.emissions
    }

    /// Take the recording apart, having proved the pools it names are the live
    /// ones.
    ///
    /// Returns the pipelines it was keeping alive; dropping them is what
    /// releases them, and doing it here rather than in `Drop` is what makes
    /// the pool check happen first.
    ///
    /// # Errors
    ///
    /// [`Mismatch`] with the retirement returned untouched and no handle read.
    pub fn release(self, pools: &WorkerPools) -> Result<Vec<Variant<V>>, (Self, Mismatch)> {
        if let Err(mismatch) = self.held.resolve(pools) {
            return Err((self, mismatch));
        }
        Ok(self.held.variants)
    }
}

/// The in-flight recordings of one device epoch, and the timeline readings
/// that retire them.
///
/// One per epoch, and it says so: a recording from a previous device names
/// pools that were destroyed with it, so [`Retirements::register`] refuses it
/// rather than holding it until a timeline value that this epoch's semaphore
/// will reach for entirely unrelated reasons.
#[derive(Debug)]
pub struct Retirements<V> {
    epoch: DeviceEpoch,
    in_flight: Vec<InFlight<V>>,
}

impl<V> Retirements<V> {
    #[must_use]
    pub const fn new(epoch: DeviceEpoch) -> Self {
        Self {
            epoch,
            in_flight: Vec::new(),
        }
    }

    #[must_use]
    pub const fn epoch(&self) -> DeviceEpoch {
        self.epoch
    }

    #[must_use]
    pub fn outstanding(&self) -> usize {
        self.in_flight.len()
    }

    /// Watch a recording until the GPU passes its point.
    ///
    /// # Errors
    ///
    /// [`Mismatch::WrongEpoch`] with the recording returned intact and nothing
    /// registered.
    pub fn register(&mut self, recording: InFlight<V>) -> Result<(), (InFlight<V>, Mismatch)> {
        if recording.held.epoch != self.epoch {
            let mismatch = Mismatch::WrongEpoch {
                expected: self.epoch,
                found: recording.held.epoch,
            };
            return Err((recording, mismatch));
        }
        self.in_flight.push(recording);
        Ok(())
    }

    /// Every registered recording the timeline has passed, in the order they
    /// were registered.
    ///
    /// Reads the timeline rather than a count or an age, for the same reason
    /// [`crate::pools::BufferRing::recycle`] does: the only fact that makes a
    /// pipeline safe to destroy is that the GPU is past the submission naming
    /// it.
    pub fn retire(&mut self, reached: TimelinePoint) -> Vec<Retired<V>> {
        let mut retired = Vec::new();
        let mut still = Vec::with_capacity(self.in_flight.len());
        for recording in self.in_flight.drain(..) {
            if reached.reached(recording.receipt.point()) {
                retired.push(Retired {
                    held: recording.held,
                    at: reached,
                });
            } else {
                still.push(recording);
            }
        }
        self.in_flight = still;
        retired
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pools::WorkerPool;
    use crate::variant::{Readiness, VariantFamily};
    use ash::vk::Handle;

    fn epoch(n: u64) -> DeviceEpoch {
        let mut e = DeviceEpoch::FIRST;
        for _ in 1..n {
            e = e.next();
        }
        e
    }

    fn at(n: u64) -> TimelinePoint {
        TimelinePoint(n)
    }

    /// A worker's pool, with distinguishable handles so that a test can tell
    /// which buffers a recording actually took.
    fn worker_pool(pool: u64, family: u32, depth: u64) -> WorkerPool {
        WorkerPool::adopt(
            vk::CommandPool::from_raw(pool),
            family,
            (0..depth)
                .map(|i| vk::CommandBuffer::from_raw(pool * 1000 + i + 1))
                .collect(),
        )
    }

    fn one_worker(depth: u64) -> WorkerPools {
        let mut pools = WorkerPools::new();
        pools.push(worker_pool(1, 3, depth));
        pools
    }

    /// A published variant, so a recording can name something whose liveness is
    /// observable.
    fn variant(family: &mut VariantFamily<u32, u64, ()>, key: u32, value: u64) -> Variant<u64> {
        let flight = family.begin_flight(key).expect("a fresh key");
        match family.publish(flight, Ok(value)) {
            Ok(Readiness::Ready(v)) => v,
            other => panic!("expected a ready variant, got {other:?}"),
        }
    }

    fn emission(set: usize) -> SetEmission {
        SetEmission {
            set,
            partial: false,
        }
    }

    fn prepared(pools: &mut WorkerPools, buffers: usize) -> Preparation<u64> {
        Preparation::begin(epoch(1), WorkerId(0), pools, buffers).expect("a free worker")
    }

    #[test]
    fn a_preparation_takes_its_worker_family_pool_and_buffers_as_one_value() {
        let mut pools = one_worker(4);
        let preparation = prepared(&mut pools, 2);

        assert_eq!(preparation.worker(), WorkerId(0));
        assert_eq!(preparation.family(), 3);
        assert_eq!(
            preparation.buffers(),
            [
                vk::CommandBuffer::from_raw(1001),
                vk::CommandBuffer::from_raw(1002)
            ]
        );
        assert_eq!(pools.of_worker(WorkerId(0)).unwrap().ring().recording(), 2);
    }

    #[test]
    fn a_recording_for_a_worker_that_does_not_exist_takes_nothing() {
        let mut pools = one_worker(4);
        let outcome = Preparation::<u64>::begin(epoch(1), WorkerId(7), &mut pools, 1);

        assert_eq!(
            outcome.err(),
            Some(Begin::Worker(Mismatch::UnknownWorker {
                worker: WorkerId(7),
                population: 1,
            }))
        );
        assert_eq!(pools.of_worker(WorkerId(0)).unwrap().ring().free(), 4);
    }

    #[test]
    fn a_partial_acquisition_gives_back_every_slot_it_took() {
        let mut pools = one_worker(3);
        let before = pools.of_worker(WorkerId(0)).unwrap().ring().clone();

        let outcome = Preparation::<u64>::begin(epoch(1), WorkerId(0), &mut pools, 4);

        assert!(matches!(outcome, Err(Begin::Depth(_))));
        let after = pools.of_worker(WorkerId(0)).unwrap().ring();
        // Free again, not leased to a preparation that does not exist. The
        // refusal counter is the one difference, and it is the record of the
        // attempt rather than an effect of it.
        assert_eq!(after.free(), 3);
        assert_eq!(after.recording(), 0);
        assert_eq!(before.free(), after.free());
    }

    #[test]
    fn acceptance_puts_every_slot_in_flight_at_the_receipt_point() {
        let mut pools = one_worker(4);
        let recorded = prepared(&mut pools, 2).finish();
        let mut submitter = crate::submission::Submitter::new();
        let reserved = submitter.reserve(at(9));
        let receipt = submitter.accepted(reserved).expect("in order");

        let in_flight = recorded
            .accepted(receipt, &mut pools)
            .unwrap_or_else(|(_, m)| panic!("{m}"));

        assert_eq!(in_flight.point(), at(9));
        let ring = pools.of_worker(WorkerId(0)).unwrap().ring();
        assert_eq!(ring.in_flight(), 2);
        assert_eq!(ring.free(), 2);

        // Only the timeline returns them, and only once it has passed the
        // point the receipt named.
        let pool = pools.for_worker(WorkerId(0)).expect("the worker");
        assert_eq!(pool.ring_mut().recycle(at(8)), 0);
        assert_eq!(pool.ring_mut().recycle(at(9)), 2);
        assert_eq!(pool.ring().free(), 4);
    }

    #[test]
    fn a_driver_refusal_frees_the_slots_without_waiting_for_a_point() {
        let mut pools = one_worker(2);
        let mut family = VariantFamily::<u32, u64, ()>::new();
        let mut preparation = prepared(&mut pools, 2);
        preparation.claimed(emission(0));
        preparation.named(variant(&mut family, 1, 40));

        let unwound = preparation
            .finish()
            .refused(&mut pools)
            .unwrap_or_else(|(_, m)| panic!("{m}"));

        let ring = pools.of_worker(WorkerId(0)).unwrap().ring();
        assert_eq!(ring.free(), 2);
        assert_eq!(ring.in_flight(), 0);
        assert_eq!(unwound.emissions(), [emission(0)]);
        assert_eq!(unwound.variants().len(), 1);
    }

    #[test]
    fn unwinding_a_preparation_returns_slots_and_the_emissions_nobody_else_knows_about() {
        let mut pools = one_worker(4);
        let mut preparation = prepared(&mut pools, 3);
        preparation.claimed(emission(0));
        preparation.claimed(emission(1));

        let emissions = preparation
            .unwind()
            .restore(WorkerId(0), &mut pools)
            .unwrap_or_else(|(_, m)| panic!("{m}"));

        assert_eq!(emissions, [emission(0), emission(1)]);
        assert_eq!(pools.of_worker(WorkerId(0)).unwrap().ring().free(), 4);
    }

    #[test]
    fn restoring_to_a_worker_that_does_not_exist_keeps_the_whole_unwinding() {
        let mut pools = one_worker(4);
        let mut preparation = prepared(&mut pools, 1);
        preparation.claimed(emission(0));

        let (unwound, mismatch) = preparation
            .unwind()
            .restore(WorkerId(4), &mut pools)
            .expect_err("no such worker");

        assert_eq!(
            mismatch,
            Mismatch::UnknownWorker {
                worker: WorkerId(4),
                population: 1,
            }
        );
        // Nothing was given back, so the retry has everything it needs.
        assert_eq!(unwound.emissions(), [emission(0)]);
        assert_eq!(pools.of_worker(WorkerId(0)).unwrap().ring().recording(), 1);
        let emissions = unwound
            .restore(WorkerId(0), &mut pools)
            .unwrap_or_else(|(_, m)| panic!("{m}"));
        assert_eq!(emissions, [emission(0)]);
        assert_eq!(pools.of_worker(WorkerId(0)).unwrap().ring().free(), 4);
    }

    #[test]
    fn a_recording_whose_worker_vanished_refuses_and_mutates_nothing() {
        let mut pools = one_worker(4);
        let recorded = prepared(&mut pools, 1).finish();
        // The epoch was torn down and rebuilt with no workers at all.
        let mut empty = WorkerPools::new();
        let receipt = {
            let mut submitter = crate::submission::Submitter::new();
            let reserved = submitter.reserve(at(1));
            submitter.accepted(reserved).expect("in order")
        };

        let (recorded, mismatch) = recorded
            .accepted(receipt, &mut empty)
            .expect_err("no such worker");

        assert_eq!(
            mismatch,
            Mismatch::UnknownWorker {
                worker: WorkerId(0),
                population: 0,
            }
        );
        assert_eq!(empty.population(), 0);
        // The original pools are untouched, and the recording is whole.
        assert_eq!(recorded.buffers().len(), 1);
        assert_eq!(pools.of_worker(WorkerId(0)).unwrap().ring().recording(), 1);
        let _ = recorded
            .accepted(receipt, &mut pools)
            .unwrap_or_else(|(_, m)| panic!("{m}"));
    }

    #[test]
    fn a_recording_offered_to_a_worker_on_another_family_refuses() {
        let mut pools = one_worker(4);
        let recorded = prepared(&mut pools, 1).finish();
        let mut moved = WorkerPools::new();
        moved.push(worker_pool(1, 5, 4));

        let (recorded, mismatch) = recorded.refused(&mut moved).expect_err("wrong family");

        assert_eq!(
            mismatch,
            Mismatch::WrongFamily {
                expected: 3,
                found: 5,
            }
        );
        assert_eq!(moved.of_worker(WorkerId(0)).unwrap().ring().free(), 4);
        assert_eq!(recorded.family(), 3);
    }

    #[test]
    fn a_recording_whose_pool_was_rebuilt_refuses_before_touching_a_handle() {
        let mut pools = one_worker(4);
        let recorded = prepared(&mut pools, 1).finish();
        let mut rebuilt = WorkerPools::new();
        rebuilt.push(worker_pool(2, 3, 4));

        let (recorded, mismatch) = recorded.refused(&mut rebuilt).expect_err("wrong pool");

        assert_eq!(mismatch, Mismatch::WrongPool);
        assert_eq!(rebuilt.of_worker(WorkerId(0)).unwrap().ring().free(), 4);
        assert_eq!(recorded.worker(), WorkerId(0));
    }

    fn registered(pools: &mut WorkerPools, point: u64) -> InFlight<u64> {
        let recorded = prepared(pools, 1).finish();
        let mut submitter = crate::submission::Submitter::new();
        let reserved = submitter.reserve(at(point));
        let receipt = submitter.accepted(reserved).expect("in order");
        recorded
            .accepted(receipt, pools)
            .unwrap_or_else(|(_, m)| panic!("{m}"))
    }

    #[test]
    fn a_recording_from_another_epoch_is_never_registered() {
        let mut pools = one_worker(4);
        let in_flight = registered(&mut pools, 5);
        let mut retirements = Retirements::<u64>::new(epoch(2));

        let (returned, mismatch) = retirements.register(in_flight).expect_err("a stale epoch");

        assert_eq!(
            mismatch,
            Mismatch::WrongEpoch {
                expected: epoch(2),
                found: epoch(1),
            }
        );
        assert_eq!(retirements.outstanding(), 0);
        assert_eq!(returned.epoch(), epoch(1));
        // The epoch that does own it takes it.
        let mut own = Retirements::new(epoch(1));
        own.register(returned)
            .unwrap_or_else(|(_, m)| panic!("{m}"));
        assert_eq!(own.outstanding(), 1);
    }

    #[test]
    fn a_recording_retires_only_when_the_timeline_passes_its_point() {
        let mut pools = one_worker(4);
        let mut retirements = Retirements::new(epoch(1));
        retirements
            .register(registered(&mut pools, 5))
            .unwrap_or_else(|(_, m)| panic!("{m}"));
        retirements
            .register(registered(&mut pools, 9))
            .unwrap_or_else(|(_, m)| panic!("{m}"));

        assert!(retirements.retire(at(4)).is_empty());
        assert_eq!(retirements.outstanding(), 2);

        let retired = retirements.retire(at(5));
        assert_eq!(retired.len(), 1);
        assert_eq!(retired[0].at(), at(5));
        assert_eq!(retirements.outstanding(), 1);

        assert_eq!(retirements.retire(at(100)).len(), 1);
        assert_eq!(retirements.outstanding(), 0);
    }

    #[test]
    fn a_named_pipeline_outlives_its_family_and_is_collected_only_after_release() {
        let mut pools = one_worker(4);
        let mut family = VariantFamily::<u32, u64, ()>::new();
        let mut preparation = prepared(&mut pools, 1);
        preparation.named(variant(&mut family, 1, 40));
        preparation.named(variant(&mut family, 2, 41));
        let recorded = preparation.finish();

        let mut submitter = crate::submission::Submitter::new();
        let reserved = submitter.reserve(at(6));
        let receipt = submitter.accepted(reserved).expect("in order");
        let in_flight = recorded
            .accepted(receipt, &mut pools)
            .unwrap_or_else(|(_, m)| panic!("{m}"));

        // The guest deleted the semantic pipeline while the work was in flight.
        family.retire();
        assert!(
            family.collect().is_empty(),
            "the recording still holds both"
        );

        let mut retirements = Retirements::new(epoch(1));
        retirements
            .register(in_flight)
            .unwrap_or_else(|(_, m)| panic!("{m}"));
        let mut retired = retirements.retire(at(6));
        assert_eq!(retired.len(), 1);
        let held = retired
            .pop()
            .expect("one")
            .release(&pools)
            .unwrap_or_else(|(_, m)| panic!("{m}"));
        assert_eq!(held.len(), 2);
        assert!(family.collect().is_empty(), "the caller still holds both");

        drop(held);
        let mut collected: Vec<u64> = family.collect();
        collected.sort_unstable();
        assert_eq!(collected, [40, 41]);
    }

    #[test]
    fn releasing_a_retirement_against_the_wrong_pools_returns_it_whole() {
        let mut pools = one_worker(4);
        let mut family = VariantFamily::<u32, u64, ()>::new();
        let mut preparation = prepared(&mut pools, 1);
        preparation.named(variant(&mut family, 1, 40));
        preparation.claimed(emission(0));
        let recorded = preparation.finish();
        let mut submitter = crate::submission::Submitter::new();
        let reserved = submitter.reserve(at(2));
        let receipt = submitter.accepted(reserved).expect("in order");
        let mut retirements = Retirements::new(epoch(1));
        retirements
            .register(
                recorded
                    .accepted(receipt, &mut pools)
                    .unwrap_or_else(|(_, m)| panic!("{m}")),
            )
            .unwrap_or_else(|(_, m)| panic!("{m}"));
        let retired = retirements.retire(at(2)).pop().expect("one");

        let mut rebuilt = WorkerPools::new();
        rebuilt.push(worker_pool(2, 3, 4));
        let (retired, mismatch) = retired.release(&rebuilt).expect_err("a rebuilt pool");

        assert_eq!(mismatch, Mismatch::WrongPool);
        assert_eq!(retired.worker(), WorkerId(0));
        assert_eq!(retired.emissions(), [emission(0)]);
        // Nothing was released, so the pipeline is still not collectable.
        family.retire();
        assert!(family.collect().is_empty());
        let held = retired
            .release(&pools)
            .unwrap_or_else(|(_, m)| panic!("{m}"));
        assert_eq!(held.len(), 1);
    }

    #[test]
    fn every_mismatch_names_itself() {
        for mismatch in [
            Mismatch::UnknownWorker {
                worker: WorkerId(1),
                population: 0,
            },
            Mismatch::WrongFamily {
                expected: 0,
                found: 1,
            },
            Mismatch::WrongPool,
            Mismatch::WrongEpoch {
                expected: epoch(1),
                found: epoch(2),
            },
        ] {
            assert!(mismatch.to_string().starts_with(mismatch.slug()));
            assert!(mismatch.slug().starts_with("vk_recording_"));
        }
    }

    #[test]
    #[should_panic(expected = "a recording with no command buffer is not one")]
    fn a_recording_with_no_command_buffer_is_refused() {
        let mut pools = one_worker(1);
        let _ = Preparation::<u64>::begin(epoch(1), WorkerId(0), &mut pools, 0);
    }
}
