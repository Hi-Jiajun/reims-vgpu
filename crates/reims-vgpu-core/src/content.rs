//! Where a backing's current bytes are, and what it costs to read them
//! somewhere else.
//!
//! # The two failures this exists to make impossible
//!
//! **A transfer with no version transition.** If a copy can happen without the
//! content having changed, then nothing bounds how many copies happen, and the
//! device pays for the same bytes every frame. Every transfer here is derived
//! from a replica being behind on specific bytes, and recording it makes it
//! not-behind — so the same transfer cannot be asked for twice without a write
//! in between.
//!
//! **A whole-resource copy to answer a partial question.** A guest writes part
//! of a buffer and a draw reads part of it. Whether a transfer is owed is a
//! question about bytes, and a per-resource dirty flag can only answer it by
//! copying everything. Freshness is a [`RangeSet`] per replica for exactly that
//! reason.
//!
//! # Authority is not placement
//!
//! [`Replica`] says *where the current bytes are*, which is a semantic fact:
//! the guest wrote them, or the device produced them into storage it owns.
//! It says nothing about how that storage is arranged, whether it is imported
//! or copied, or which memory type it lives in — those are an executor's
//! decisions about one capability cell, and a semantic model that knew them
//! would make the same guest stream mean different things on two hosts.

use crate::access::{BackingId, ByteRange, ContentVersion};
use crate::range_set::RangeSet;
use std::collections::HashMap;

/// Where a copy of a backing's content lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Replica {
    /// The guest's own pages.
    GuestPages,
    /// Storage this device owns. Whether that storage is an import of the
    /// guest's pages or a separate allocation is a placement decision and not
    /// this one.
    DeviceOwned,
}

impl Replica {
    pub const BOTH: [Replica; 2] = [Replica::GuestPages, Replica::DeviceOwned];

    pub const fn name(self) -> &'static str {
        match self {
            Self::GuestPages => "guest_pages",
            Self::DeviceOwned => "device_owned",
        }
    }

    #[must_use]
    pub const fn other(self) -> Replica {
        match self {
            Self::GuestPages => Self::DeviceOwned,
            Self::DeviceOwned => Self::GuestPages,
        }
    }
}

/// A copy that is owed before a read can be served.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transfer {
    pub backing: BackingId,
    pub from: Replica,
    pub to: Replica,
    /// Exactly the bytes the destination is behind on.
    pub bytes: RangeSet,
}

/// What has happened to one backing's content.
#[derive(Clone, Debug, Default)]
struct Entry {
    version: ContentVersion,
    fresh: HashMap<Replica, RangeSet>,
}

/// The content authority for one session.
#[derive(Debug, Default)]
pub struct ContentLedger {
    backings: HashMap<BackingId, Entry>,
    census: Census,
}

/// What the ledger has been asked for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    /// Writes recorded.
    pub writes: usize,
    /// Transfers the ledger said were owed.
    pub transfers_planned: usize,
    /// Bytes in those transfers.
    pub transfer_bytes: u64,
    /// Reads that needed no transfer. The number that says how much the
    /// per-byte freshness is buying over a per-resource flag.
    pub reads_already_fresh: usize,
}

impl ContentLedger {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    /// The backing's current content version.
    #[must_use]
    pub fn version(&self, backing: BackingId) -> ContentVersion {
        self.backings
            .get(&backing)
            .map_or(ContentVersion::default(), |e| e.version)
    }

    /// Declare that a backing's content originates in one replica and is
    /// entirely current there.
    ///
    /// This is creation, not a write: a resource whose pages the guest supplied
    /// starts with the guest fresh for all of it and the device owning nothing.
    /// Calling it on a live backing resets the authority, which is what a
    /// replace-physical does and what nothing else may do.
    pub fn declare(&mut self, backing: BackingId, extent: ByteRange, authority: Replica) {
        let e = self.backings.entry(backing).or_default();
        e.version = e.version.next();
        e.fresh.clear();
        e.fresh.insert(authority, RangeSet::from_range(extent));
    }

    /// Record a write of `bytes` performed in `replica`.
    ///
    /// The version advances once per write, and the writing replica becomes
    /// the only one fresh for those bytes. Every other replica loses them,
    /// which is what makes a later read from elsewhere owe a transfer — and
    /// what makes a read from *here* owe nothing.
    pub fn write(&mut self, backing: BackingId, bytes: ByteRange, replica: Replica) {
        self.census.writes += 1;
        let e = self.backings.entry(backing).or_default();
        e.version = e.version.next();
        for r in Replica::BOTH {
            let set = e.fresh.entry(r).or_default();
            if r == replica {
                set.insert(bytes);
            } else {
                set.remove(bytes);
            }
        }
    }

    /// The transfer a read of `bytes` from `replica` needs, if any.
    ///
    /// `None` means the replica already holds those bytes at the current
    /// version. It is not a promise that the read is free — that depends on
    /// placement — only that no content has to move for it to be correct.
    #[must_use]
    pub fn transfer_for_read(
        &mut self,
        backing: BackingId,
        bytes: ByteRange,
        replica: Replica,
    ) -> Option<Transfer> {
        let e = self.backings.get(&backing)?;
        let fresh = e.fresh.get(&replica).cloned().unwrap_or_default();
        let owed = fresh.missing_from(bytes);
        if owed.is_empty() {
            self.census.reads_already_fresh += 1;
            return None;
        }
        // Only the other replica can supply them, and only the parts it is
        // itself fresh for. Bytes neither replica holds have never been
        // written, so there is nothing to move and no version to preserve —
        // copying them would move whatever the source happens to contain and
        // then claim the destination is current.
        let source = replica.other();
        let source_fresh = e.fresh.get(&source).cloned().unwrap_or_default();
        let mut movable = RangeSet::new();
        for r in owed.ranges() {
            for s in source_fresh.ranges() {
                if let Some(overlap) = intersect(*r, *s) {
                    movable.insert(overlap);
                }
            }
        }
        if movable.is_empty() {
            return None;
        }
        self.census.transfers_planned += 1;
        self.census.transfer_bytes += movable.len();
        Some(Transfer {
            backing,
            from: source,
            to: replica,
            bytes: movable,
        })
    }

    /// Record that a planned transfer has completed.
    ///
    /// The destination becomes fresh for exactly the bytes that moved, and the
    /// source keeps what it had: a copy does not change where content is
    /// authoritative, only how many places hold it.
    pub fn record_transfer(&mut self, transfer: &Transfer) {
        let e = self.backings.entry(transfer.backing).or_default();
        e.fresh
            .entry(transfer.to)
            .or_default()
            .union_with(&transfer.bytes);
    }

    /// Discard a replica's copy of some bytes without changing the content.
    ///
    /// This is what a discard packet means: the bytes are still whatever they
    /// were, and this replica no longer holds them. The version does not
    /// advance, because nothing was written.
    ///
    /// Region-scoped rather than whole-backing, because several resources share
    /// one backing whenever they are placed in one heap, and discarding a
    /// heap-placed resource's copy must not throw away its neighbours'.
    pub fn discard(&mut self, backing: BackingId, bytes: ByteRange, replica: Replica) {
        if let Some(e) = self.backings.get_mut(&backing) {
            if let Some(set) = e.fresh.get_mut(&replica) {
                set.remove(bytes);
            }
        }
    }

    /// The bytes of `range` that only `replica` holds.
    ///
    /// The question a discard has to ask before it takes a hint: content no
    /// other replica is fresh for exists nowhere else, so dropping it is not a
    /// memory saving but a loss of bytes the guest may still read.
    #[must_use]
    pub fn sole_authority(
        &self,
        backing: BackingId,
        range: ByteRange,
        replica: Replica,
    ) -> RangeSet {
        let mut out = RangeSet::new();
        let Some(e) = self.backings.get(&backing) else {
            return out;
        };
        if let Some(here) = e.fresh.get(&replica) {
            for r in here.ranges() {
                if let Some(overlap) = intersect(*r, range) {
                    out.insert(overlap);
                }
            }
        }
        if let Some(elsewhere) = e.fresh.get(&replica.other()) {
            for r in elsewhere.ranges() {
                out.remove(*r);
            }
        }
        out
    }

    /// Forget a backing entirely.
    pub fn forget(&mut self, backing: BackingId) {
        self.backings.remove(&backing);
    }

    /// Whether a replica is fresh for every byte of a range.
    #[must_use]
    pub fn is_fresh(&self, backing: BackingId, bytes: ByteRange, replica: Replica) -> bool {
        self.backings
            .get(&backing)
            .and_then(|e| e.fresh.get(&replica))
            .is_some_and(|set| set.covers(bytes))
    }
}

fn intersect(a: ByteRange, b: ByteRange) -> Option<ByteRange> {
    let start = a.offset.max(b.offset);
    let end = a
        .offset
        .saturating_add(a.length)
        .min(b.offset.saturating_add(b.length));
    (start < end).then(|| ByteRange {
        offset: start,
        length: end - start,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(offset: u64, length: u64) -> ByteRange {
        ByteRange { offset, length }
    }
    const B: BackingId = BackingId(1);

    #[test]
    fn a_declared_backing_is_fresh_where_it_was_declared_and_nowhere_else() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 256), Replica::GuestPages);
        assert!(c.is_fresh(B, r(0, 256), Replica::GuestPages));
        assert!(!c.is_fresh(B, r(0, 1), Replica::DeviceOwned));
        assert_eq!(c.version(B), ContentVersion(1));
    }

    /// The first failure this module exists to prevent: a copy without a
    /// version transition. Asking twice for the same read cannot produce two
    /// transfers.
    #[test]
    fn a_transfer_recorded_is_a_transfer_not_asked_for_again() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 256), Replica::GuestPages);
        let t = c
            .transfer_for_read(B, r(0, 64), Replica::DeviceOwned)
            .expect("the device holds nothing yet");
        assert_eq!(t.from, Replica::GuestPages);
        assert_eq!(t.bytes.ranges(), &[r(0, 64)]);
        c.record_transfer(&t);
        assert!(c
            .transfer_for_read(B, r(0, 64), Replica::DeviceOwned)
            .is_none());
        assert_eq!(c.census().transfers_planned, 1);
        assert_eq!(c.census().reads_already_fresh, 1);
    }

    /// And a write is what makes it owed again.
    #[test]
    fn a_write_elsewhere_makes_the_transfer_owed_again() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 256), Replica::GuestPages);
        let t = c
            .transfer_for_read(B, r(0, 64), Replica::DeviceOwned)
            .unwrap();
        c.record_transfer(&t);
        let before = c.version(B);
        c.write(B, r(0, 16), Replica::GuestPages);
        assert!(c.version(B) > before, "a write is a version transition");
        let again = c
            .transfer_for_read(B, r(0, 64), Replica::DeviceOwned)
            .expect("the first sixteen bytes moved under it");
        assert_eq!(
            again.bytes.ranges(),
            &[r(0, 16)],
            "only the bytes that changed are owed, not the whole read"
        );
    }

    /// The second failure: answering a partial question with a whole-resource
    /// copy.
    #[test]
    fn a_partial_read_owes_only_the_bytes_it_is_behind_on() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 4096), Replica::GuestPages);
        let t = c
            .transfer_for_read(B, r(0, 4096), Replica::DeviceOwned)
            .unwrap();
        c.record_transfer(&t);
        c.write(B, r(1024, 16), Replica::GuestPages);
        c.write(B, r(3000, 8), Replica::GuestPages);
        let owed = c
            .transfer_for_read(B, r(0, 4096), Replica::DeviceOwned)
            .unwrap();
        assert_eq!(owed.bytes.ranges(), &[r(1024, 16), r(3000, 8)]);
        assert_eq!(owed.bytes.len(), 24, "24 bytes, not 4096");
    }

    /// A read from the replica that just wrote owes nothing, whatever else has
    /// happened to the backing.
    #[test]
    fn a_read_from_the_writer_is_always_fresh() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 256), Replica::GuestPages);
        c.write(B, r(0, 256), Replica::DeviceOwned);
        assert!(c
            .transfer_for_read(B, r(0, 256), Replica::DeviceOwned)
            .is_none());
        let back = c
            .transfer_for_read(B, r(0, 256), Replica::GuestPages)
            .expect("the guest lost them when the device wrote");
        assert_eq!(back.from, Replica::DeviceOwned);
    }

    /// Bytes neither replica has ever held are not a transfer. Copying them
    /// would move whatever the source happens to contain and then claim the
    /// destination is current — which is worse than leaving both undefined,
    /// because the second one is honest.
    #[test]
    fn bytes_nobody_has_written_are_not_copied() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 64), Replica::GuestPages);
        let t = c
            .transfer_for_read(B, r(0, 256), Replica::DeviceOwned)
            .expect("the declared part is owed");
        assert_eq!(
            t.bytes.ranges(),
            &[r(0, 64)],
            "only the declared extent moves; the rest was never anyone's"
        );
    }

    /// A discard drops a copy and not the content, so the version does not
    /// move and the other replica is still authoritative.
    #[test]
    fn a_discard_loses_a_copy_and_not_the_content() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 256), Replica::GuestPages);
        let t = c
            .transfer_for_read(B, r(0, 256), Replica::DeviceOwned)
            .unwrap();
        c.record_transfer(&t);
        let version = c.version(B);
        c.discard(B, r(0, 256), Replica::DeviceOwned);
        assert_eq!(c.version(B), version, "nothing was written");
        assert!(c.is_fresh(B, r(0, 256), Replica::GuestPages));
        assert!(c
            .transfer_for_read(B, r(0, 256), Replica::DeviceOwned)
            .is_some());
    }

    /// Replace-physical re-declares: the guest re-pointed the resource, so
    /// every copy anyone held is of memory this resource no longer names.
    #[test]
    fn redeclaring_a_backing_invalidates_every_copy() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 256), Replica::GuestPages);
        let t = c
            .transfer_for_read(B, r(0, 256), Replica::DeviceOwned)
            .unwrap();
        c.record_transfer(&t);
        c.declare(B, r(0, 256), Replica::GuestPages);
        assert!(!c.is_fresh(B, r(0, 1), Replica::DeviceOwned));
        assert_eq!(
            c.version(B),
            ContentVersion(2),
            "two declarations; the transfer between them was not a version \
             transition, which is the whole reason it could not repeat"
        );
    }

    #[test]
    fn a_forgotten_backing_answers_nothing() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 256), Replica::GuestPages);
        c.forget(B);
        assert!(c
            .transfer_for_read(B, r(0, 256), Replica::DeviceOwned)
            .is_none());
        assert_eq!(c.version(B), ContentVersion::default());
    }
    /// Two resources placed in one heap share a backing, so a discard has to
    /// be about bytes. A whole-backing discard would throw away a neighbour's
    /// copy and charge it for a transfer it did not earn.
    #[test]
    fn discarding_one_placement_leaves_its_neighbour_fresh() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 512), Replica::GuestPages);
        let t = c
            .transfer_for_read(B, r(0, 512), Replica::DeviceOwned)
            .unwrap();
        c.record_transfer(&t);
        c.discard(B, r(0, 256), Replica::DeviceOwned);
        assert!(!c.is_fresh(B, r(0, 256), Replica::DeviceOwned));
        assert!(c.is_fresh(B, r(256, 256), Replica::DeviceOwned));
    }

    /// The question a discard asks before taking a hint. Content only one
    /// replica holds is not a spare copy.
    #[test]
    fn sole_authority_is_the_bytes_nowhere_else_holds() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 512), Replica::GuestPages);
        assert!(
            c.sole_authority(B, r(0, 512), Replica::DeviceOwned)
                .is_empty(),
            "the device holds nothing yet"
        );
        assert_eq!(
            c.sole_authority(B, r(0, 512), Replica::GuestPages).ranges(),
            &[r(0, 512)],
            "and the guest holds all of it alone"
        );
        let t = c
            .transfer_for_read(B, r(0, 512), Replica::DeviceOwned)
            .unwrap();
        c.record_transfer(&t);
        assert!(
            c.sole_authority(B, r(0, 512), Replica::GuestPages)
                .is_empty(),
            "a copy is a second holder"
        );
        // A device write of part of it makes that part the device's alone.
        c.write(B, r(128, 64), Replica::DeviceOwned);
        assert_eq!(
            c.sole_authority(B, r(0, 512), Replica::DeviceOwned)
                .ranges(),
            &[r(128, 64)]
        );
        assert_eq!(
            c.sole_authority(B, r(0, 128), Replica::DeviceOwned)
                .ranges(),
            &[],
            "and the question is bounded by the range asked about"
        );
    }
}
