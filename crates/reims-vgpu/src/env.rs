//! Every environment variable this device reads, and the one way they parse.
//!
//! # Why they all live here
//!
//! An override is a rule the operator states from outside the process, so it has
//! the same problem the ABI header has: nothing in the toolchain finds the second
//! copy. A variable read at its point of use is invisible to everyone who does not
//! already know it exists, two sites spelling one variable's "off" differently is
//! a divergence no test can see, and a name that gets renamed in one place keeps
//! working in the other. Naming them here makes the set greppable and makes the
//! parse shared.
//!
//! # What an override may do
//!
//! **An override may only narrow what this device does. It may never widen it.**
//!
//! A switch can turn a rail *off* that the host was capable of running, because
//! that is a statement about policy and is always satisfiable. A switch may not
//! turn a rail *on* that the host reported it cannot run: capability is measured
//! from the device, and a variable that could override the measurement would turn
//! "this host has no such extension" into a crash or, worse, undefined behavior
//! inside a driver. Every gate stays where it is; a switch can only add a reason
//! to refuse.
//!
//! That rule is why [`Switch::On`] exists but is nowhere sufficient on its own.
//! Reading it is how a caller notices an operator asked for something the host
//! cannot give and says so, rather than ignoring the request in silence.

/// Guest RAM reaches the GPU as a host-pointer import over whole RAMBlocks.
/// Setting this off makes the device take the copying rails on a host that
/// could have imported — see
/// [`crate::backend::vulkan::caps::host_pointer`].
///
/// This is the switch that matters for verification. Where the import works
/// every guest window takes it and the copying rails run zero times, so a green
/// boot says nothing about them — and they are the only rails on a host without
/// the extension, and the rails a discrete GPU takes regardless.
pub const GUEST_IMPORT: &str = "REIMS_VGPU_GUEST_IMPORT";

/// Verbose per-draw logging on top of the always-on fail sink.
pub const DRAW_LOG: &str = "REIMS_VGPU_DRAW_LOG";

/// Setting this off makes a completion stamp that follows a guest-page writeback
/// block the drain worker on that writeback and then write the stamp word
/// itself, instead of recording the word into the same GPU queue behind the
/// copy and letting the completion thread raise the interrupt.
///
/// A narrowing, like every switch here: the GPU-ordered stamp needs a
/// host-pointer import to reach the stamp page and `timelineSemaphore` to be
/// waited off-thread, so `off` selects the rail a host lacking either takes
/// regardless. It exists because the two rails answer "when may the guest
/// observe this stamp" with different mechanisms — a CPU wait versus a pipeline
/// barrier plus a thread — and a hang or a torn frame has to be attributable to
/// one of them without rebuilding.
pub const GPU_STAMP: &str = "REIMS_VGPU_GPU_STAMP";

/// Setting this off stops the two guest-page write guards —
/// [`crate::runtime::node_guard`] and [`crate::runtime::released_pages`] — from
/// observing anything. They decide nothing, so this changes no guest-visible
/// behavior; what it removes is the page-table descent and the page-list resolve
/// that each map and unmap packet pays for them, on the drain thread, while it
/// holds the device lock.
///
/// A narrowing, like every switch here: it turns an observation off and can
/// never turn one on.
///
/// It exists because these guards watch an intermittent guest kernel panic that
/// is a **race**, so the honest question "does watching it change the rate?" has
/// to be answerable without rebuilding. A measurement that cannot be controlled
/// is the failure this whole instrument was built to avoid, and an instrument on
/// the drain thread is exactly the kind that could perturb its own subject.
pub const PAGE_GUARDS: &str = "REIMS_VGPU_PAGE_GUARDS";

/// Setting this **on** makes [`crate::runtime::range_coverage`] walk the guest's
/// page table across every page of every map and unmap range. Default off, and
/// it is the only variable here whose default is the quiet one.
///
/// # Why it defaults off, which is a measurement and not a preference
///
/// Always-on, this walk costs the drain enough to lose the guest a race it
/// otherwise wins. One undriven macos-15 boot went from **0** `no_list_entry` to
/// **47**, and from 0 `list_miss_slot_empty` to 182, purely by adding it — and
/// back to 0 with the guards switched off on the same binary. The other guards
/// descend a single path per packet; this one walks a whole range, sixteen
/// thousand pages for one 64 MiB mapping, on the drain thread while it holds the
/// device lock.
///
/// So it is a probe rather than a guard, under the rule its own module states:
/// an instrument that watches a race must not be the reason the race moves.
///
/// # Why it is not the widening this module forbids
///
/// The rule above is about **capability** — a switch may not turn on a rail the
/// host reported it cannot run, because binding an unadvertised extension is a
/// crash and importing an undeclared handle is undefined behavior in a driver.
/// There is no host that cannot walk a page table it is already reading, and
/// nothing this gates changes what the guest observes. What it changes is how
/// much work the drain does, and the default is the side that does less.
///
/// It gets its own name rather than riding on [`DRAW_LOG`] for the same reason
/// it exists: that variable turns on a per-draw log flood that is itself a drain
/// cost, so gating a latency probe behind it would guarantee the perturbation
/// the probe is trying to measure.
pub const RANGE_COVERAGE: &str = "REIMS_VGPU_RANGE_COVERAGE";

/// `off` stops narrowing a guest buffer bind to the extent the shader's
/// reflection proved it can read, so the bind walks the rest of the allocation
/// exactly as it did before that rail existed.
///
/// This is the A/B instrument for the rail, and it is why the rail can be
/// measured at all: the two arms differ by one branch in one process, so a
/// driven boot of each on one build and one rail attributes a change in gathered
/// bytes to the narrowing rather than to a rebuild. Without it the comparison is
/// a boot of `HEAD` against a boot of `HEAD~1`, which also moves every other
/// difference between the two binaries into the result.
///
/// It only ever *widens the window this device reads*, never what the guest may
/// see, so it obeys the rule the module doc states: it turns a rail off, and
/// there is no spelling of it that turns one on. `on` and unset are the same
/// arm — the default — because a capability that is not measured is not a
/// capability this switch may grant.
pub const BUFFER_EXTENT: &str = "REIMS_VGPU_BUFFER_EXTENT";

/// `off` narrows the draw batch back to one render target, so a draw whose
/// target differs from the open batch's stops joining it and submits its own
/// command buffer.
///
/// The wider arm — the default — is that the target does not key the batch at
/// all: every batched draw begins and ends its own render pass inside the
/// command buffer, and nothing between `batch_append` and `batch_flush` reads
/// which image those passes wrote. A run alternating between two surfaces
/// therefore costs one submission per draw under the narrow arm and one per
/// `BATCH_MAX_DRAWS` draws under the wide one.
///
/// It exists as a switch for the same reason [`BUFFER_EXTENT`] does: the two
/// arms differ by one comparison in one process, so a driven boot of each on one
/// build and one guest rail attributes a change in submissions, ring blocking
/// and gathered bytes to the batching rule rather than to a rebuild. Off is a
/// refusal (`nojoin_target_switch`) and never a permission.
pub const BATCH_MIXED_TARGETS: &str = "REIMS_VGPU_BATCH_MIXED_TARGETS";

/// `off` stages the guest bytes of a `[[buffer(n)]]` bind even when the stage's
/// own reflection says the shader never dereferences it.
///
/// The wider arm — the default — binds a neutral page for such a bind instead of
/// gathering the guest's, because
/// [`crate::runtime::spirv_bind::ReflectedBufferAccess::Unused`] means no shader
/// invocation reads through the descriptor. The descriptor is still written, so
/// the pipeline layout is byte-for-byte what it was; only the contents change,
/// and only for binds nothing reads.
///
/// It exists as a switch for the same reason [`BUFFER_EXTENT`] does, and with
/// more at stake. This is the one rail here whose failure mode is **silent wrong
/// pixels**: if the translator ever says `Unused` about a buffer a shader does
/// dereference, the shader reads the neutral page and nothing anywhere reports
/// it. So the arm that copies has to stay reachable in one process, both to A/B
/// the saving and to answer "is this rail why that surface is wrong" without a
/// rebuild.
///
/// Off is a refusal and never a permission: it makes this device read *more* of
/// the guest's memory, never less, and there is no spelling that grants a
/// capability.
///
/// # It buys frames now, and did not when it was written
///
/// The twelve-boot A/B that landed this rail established the saving —
/// `kib_per_draw` −6.27 %, disjoint at 45x — and explicitly established **no
/// frame gain**, correctly, because the host window presenter was then a hard
/// ~41 Hz ceiling and no device-side saving could appear past it.
///
/// Re-run on the multi-flight presenter, eight driven macos-13 boots, n=3 vs
/// n=2 after regime exclusion: `kib_per_draw` **−6.31 %** — the same number to
/// within a twentieth of a percent, disjoint at 44x, which is what says the two
/// runs measured one rail — and now `frames_s` **+3.73 %, disjoint**, with
/// `draws_s` +2.93 % also disjoint.
///
/// So the saving was real all along and was being spent into a ceiling. Two
/// readings follow, and the second is the one worth carrying: a per-draw saving
/// measured before that presenter fix is owed a re-run rather than believed
/// worthless, and `frames_s` is the metric that reports it — not `presents_s`,
/// which still overlapped here at this sample size.
pub const UNUSED_BINDS: &str = "REIMS_VGPU_UNUSED_BINDS";

/// `off` returns the host window presenter to one present in flight at a time.
///
/// The wider arm — the default — lets several of the presenter's blits be in
/// flight at once, because with one the presenter was a **ceiling** rather than
/// a pacer: twelve driven macos-13 boots put its output at 1599-1696 frames
/// while the device published 1760-2015 to it, `busy_acquire` 0 throughout. The
/// swapchain always had an image free; the refusals were all the previous
/// blit's fence, which retires behind queued guest work because the blit shares
/// a queue with every guest draw.
///
/// It exists as a switch for the same reason [`BUFFER_EXTENT`] does — one
/// binary, one branch, two arms — and because presentation depth is the kind of
/// change whose failure is a stutter or a torn frame rather than a decline, so
/// the previous behavior has to stay reachable without a rebuild.
///
/// Off is a refusal and never a permission: one present in flight is strictly
/// less concurrency than several, never more.
pub const PRESENT_DEPTH: &str = "REIMS_VGPU_PRESENT_DEPTH";

/// **Probe, default off.** Setting this *on* cuts every guest-page scatter run
/// into four contiguous sub-ranges that tile it exactly.
///
/// The guest bytes written are byte-for-byte identical either way — only the
/// number of `VkBufferCopy` regions changes, by 4x. It is the controlled form of
/// the question the writeback rail's cost turns on: whether that rail is bound
/// by the bytes it moves or by the number of copy regions it issues. The two
/// predict opposite things about replacing the scatter with a compute dispatch,
/// and a host GPU at 86-91 % busy on 3-4 % memory utilization says it is not the
/// bytes.
///
/// It is a probe and not a rail, in the sense [`RANGE_COVERAGE`] is: it changes
/// nothing the guest observes, and its default is the side that does less work.
/// It does not widen anything — there is no host that can issue one copy region
/// and not four.
///
/// # What it measured, and why it is kept rather than deleted
///
/// Eight driven macos-13 boots, four per arm, one binary: 203 regions per
/// writeback against 806, and `present_hz` **49.15/49.45/56.45/56.40 against
/// 26.90/23.80/23.00/23.70**. Eight boots for eight with no overlap — four times
/// the regions for byte-identical output **halves the frame rate**, and
/// `slot_us` roughly doubles, which is the drain worker blocking longer on a ring
/// fence the GPU takes longer to signal.
///
/// That answers the question for this host class and it is written up where the
/// rail is, in [`crate::runtime::render_writeback`]. The probe stays because the
/// answer is a property of the **host**, not of this device: a discrete GPU
/// crossing PCIe per region and a unified-memory host writing into the same
/// physical pages have no reason to agree, and only one of the two has been
/// measured. A future unified-memory boot re-runs this in one command instead of
/// rebuilding the experiment from the module doc.
pub const SCATTER_SPLIT: &str = "REIMS_VGPU_SCATTER_SPLIT";

/// `off` narrows the guest-page writeback's scatter back to one transfer region
/// per guest run, from the compute dispatch that replaces them.
///
/// The dispatch writes the same guest bytes — the kernel copies `uint`s and
/// carries no format, row or texel semantics at all — so this switch chooses
/// between two byte-identical implementations of one copy and can never change
/// what the guest observes. It narrows in the sense the module doc requires:
/// the transfer form is the only form on a host without the guest-RAM import,
/// and it stays the form for a run whose geometry the dispatch cannot express.
///
/// It exists because it is the A/B. The region count is measured to be ~35 % of
/// frame time (see [`crate::runtime::render_writeback`]), and the only way to
/// hold that number against this repair on a given host is to run the host both
/// ways in one binary.
pub const COMPUTE_SCATTER: &str = "REIMS_VGPU_COMPUTE_SCATTER";

/// `off` narrows a draw chain's pipeline resolution back to the full walk —
/// object list, descriptor, decode, MTLB read, AIR carve and content hash, for
/// the pipeline and both of its functions, on every draw.
///
/// It narrows in the sense this module requires: the full walk is what the memo
/// is a cache in front of, and every resolution the memo serves came out of it.
/// Switching it off cannot reach a resolution the walk would not have produced.
///
/// It exists because the memo's correctness rests on a stated claim about what a
/// guest does to a live pipeline object — see
/// [`crate::runtime::pipeline_resolve`] — and a claim about a guest is worth a
/// binary that can be run both ways against that guest.
pub const PIPELINE_MEMO: &str = "REIMS_VGPU_PIPELINE_MEMO";

/// `on` issues the draw-time guest buffer gather as one compute dispatch per
/// gathered window instead of one transfer region per guest run.
///
/// The gather direction of what [`COMPUTE_SCATTER`] does for the writeback, and
/// byte-identical for the same reason: the kernel copies `uint`s and carries no
/// format, row or direction semantics at all, and the run table is built from
/// the very `VkBufferCopy` regions the transfer form would have issued. So this
/// chooses between two implementations of one copy and can never change what
/// the guest observes.
///
/// # It is default **off**, and this is the measurement that says so
///
/// Ten driven macos-13 sustained-animation boots. Comparing only the four that
/// landed in the same compositing sub-regime (draws/frame ~351, so like for
/// like), the dispatch does exactly what it was built to do to the GPU and more
/// than that to the CPU:
///
/// ```text
///                     on (n=2)        off (n=2)
/// slot_us          48 838 / 62 875  124 703 / 130 423   -56 %  GPU blocking
/// ring_retire_blocks          260              502      -48 %
/// record_us        91 961 / 93 106   49 458 /  50 562   +85 %  CPU recording
/// descriptors_us   10 387 /  9 109    6 796 /   6 757   +45 %
/// frames/s          74.68 /  75.64    76.04 /   76.12    -1.4 %
/// ```
///
/// So the mechanism works — halving the blocking on a rail this device is
/// **GPU-bound** on is exactly the lever — and the implementation gives it all
/// back. The reason is the count: ~40 000 dispatches a second, one per gathered
/// window, each paying an `acquire_staging` for its run table, a
/// `write_staging`, a descriptor-set allocation and an `update_descriptor_sets`.
/// The writeback's scatter issues ~1 900 a second and does not notice any of
/// that.
///
/// # What would flip it, and what would not
///
/// **Not fewer dispatches.** The obvious move is to batch a command buffer's
/// gathers into one, and the arithmetic refuses it: ~40 000 dispatches against
/// ~26 500 draws is **1.5 per draw**, not 18 per command buffer. A gather is
/// recorded ahead of its own draw's render pass and cannot be hoisted to the
/// front of the command buffer, because a Store earlier in that same command
/// buffer may have landed in the pages it reads — which is exactly why
/// `ResourcePools::note_guest_write_recorded` invalidates the bind map. So
/// batching buys at most 1.5x on the count.
///
/// **The per-dispatch cost, which is ~1.05 us of the 1.6 us a draw this
/// regressed by.** Three things are paid per dispatch and all three are
/// avoidable:
///
/// - an `acquire_staging` and a `write_staging` for a ~200-byte run table.
///   One command-buffer-scoped arena for run tables, with each dispatch reading
///   at its own offset, makes that ~2 200 acquisitions a second instead of
///   40 000.
/// - a descriptor-set allocation and an `update_descriptor_sets`. Two of the
///   three bindings — the guest import and the run-table arena — are the *same
///   buffer* for every dispatch in a command buffer. Only `Dst` differs, and
///   only because each gathered window takes its own pooled slot. Suballocate
///   those from one command-buffer arena instead and all three bindings become
///   constant, so the whole command buffer needs **one** descriptor set.
///   `BoundBuffer` already carries an offset, so the bind side of that is free.
/// - the bind and push. With one set per command buffer, a dispatch is a push
///   constant carrying its run-table base index and `vkCmdDispatch`.
///
/// Together that is ~0.05 us a dispatch against today's ~1.05, holding a GPU
/// saving already measured at 56 %. That is the change worth making; a first
/// pass at only the count is measurably not.
///
/// # A region-count threshold is not the shortcut it looks like
///
/// The dispatch's value scales with how many regions each one replaces, so the
/// obvious cheap interim is to dispatch only for windows above some run count.
/// Six driven boots of `gpu-load-probe` at `layers=24&boxes=6`, which runs
/// **23.5 gather regions per draw** against the sustained probe's 15.8, say the
/// ceiling on that is low. Comparing the boots that reached the same drain duty
/// (~0.8):
///
/// ```text
///            frames/s   slot_us   record_us
/// on               86.96    13 007      64 357
/// off              84.76    95 715      42 487
/// off              83.72   107 146      42 279
/// ```
///
/// The sign does flip — `slot_us` falls **86 %** here against 56 % on the
/// lighter load, exactly as amortising a fixed cost over more regions predicts
/// — and it is still only ~+3 % of the frames, because the per-dispatch cost is
/// unchanged and the recording penalty is the same +52 %. A threshold buys the
/// tail of a distribution whose mean is the problem.
///
/// That load is also a poor A/B vehicle and should not be used as one: the same
/// six boots ranged over drain duty 0.25 to 0.80 and 24.7 to 87.0 frames a
/// second, while `draws/frame` sat at 132.7-133.0 on every one of them. The
/// regime discriminator that works on the sustained probe is flat here, so
/// nothing separates a fast boot from a slow one before the fact.
///
/// Until then the switch is a permission rather than a refusal, which is the
/// one place this module's own rule is bent. It is bent knowingly: that rule is
/// about **host capability** — binding an unadvertised extension crashes and
/// importing an undeclared handle is undefined behaviour — and neither arm here
/// asks the host for anything the other does not. What differs is only which of
/// two byte-identical copies runs, and the default is the one that measured
/// faster.
pub const COMPUTE_GATHER: &str = "REIMS_VGPU_COMPUTE_GATHER";

/// What one variable says, including the two ways it says nothing usable.
///
/// Four states rather than a `bool` because "unset", "explicitly on" and
/// "spelled wrong" are three different operator intents and a `bool` collapses
/// them into the default. The last one matters most: a typo that silently reads
/// as the default is how an operator concludes a switch does not work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Switch {
    /// Not in the environment, or exported empty — which is how a shell says
    /// "not set" when a variable is assigned from an unset variable.
    Unset,
    /// An affirmative spelling. Never sufficient by itself; see the module doc.
    On,
    /// A negative spelling. This is the state that may change behavior.
    Off,
    /// Present, non-empty, and not one of the spellings below. Carries nothing:
    /// the value is handed back by [`read`] for the caller to name in its own
    /// refusal, because only the caller knows which variable this was.
    Unrecognized,
}

/// The spellings accepted for each state, ASCII-case-insensitively.
///
/// The conventional shell set rather than a chosen one, so an operator does not
/// have to look up which of `0`/`false`/`no` this particular program wanted. The
/// two lists are disjoint and every entry is lowercase, which
/// `the_spellings_are_disjoint_and_lowercase` pins.
const ON_SPELLINGS: [&str; 4] = ["1", "on", "true", "yes"];
const OFF_SPELLINGS: [&str; 4] = ["0", "off", "false", "no"];

/// Classify `name`'s value, and hand back the raw value for a caller that needs
/// to quote it.
///
/// Pure: it reads the environment and parses, and emits nothing. Deliberately —
/// [`crate::observe`] itself reads a variable through here, so an emit on this
/// path would recurse through the sink that is asking whether it is enabled.
/// The caller emits, and it is better placed to: it knows which rail the answer
/// gates and what the consequence of refusing is.
pub fn read(name: &str) -> (Switch, Option<String>) {
    let Some(raw) = std::env::var_os(name) else {
        return (Switch::Unset, None);
    };
    let value = raw.to_string_lossy().into_owned();
    let folded = value.trim().to_ascii_lowercase();
    if folded.is_empty() {
        return (Switch::Unset, None);
    }
    let state = if ON_SPELLINGS.contains(&folded.as_str()) {
        Switch::On
    } else if OFF_SPELLINGS.contains(&folded.as_str()) {
        Switch::Off
    } else {
        Switch::Unrecognized
    };
    (state, Some(value))
}

/// [`read`] for a caller that has nothing to say about the value.
pub fn switch(name: &str) -> Switch {
    read(name).0
}

/// Every variable this device reads.
///
/// The one place the set is enumerable. A boot line built from this reports what
/// an operator actually set, which is the difference between a bug report that
/// says "it is slow" and one that says "it is slow with a rail switched off" —
/// and an operator who mistyped a value learns it from the same line, because
/// [`Switch::Unrecognized`] has its own spelling here.
///
/// Nothing enforces that a new `pub const` above is added to this list; the rule
/// is stated and honestly unenforced. What keeps it small is that the list is
/// next to the constants, and [`report_line`] is the only consumer.
pub const ALL: [&str; 11] = [
    GUEST_IMPORT,
    DRAW_LOG,
    GPU_STAMP,
    PAGE_GUARDS,
    RANGE_COVERAGE,
    BUFFER_EXTENT,
    BATCH_MIXED_TARGETS,
    UNUSED_BINDS,
    PRESENT_DEPTH,
    SCATTER_SPLIT,
    COMPUTE_SCATTER,
];

/// The state of every variable in [`ALL`], for the one-shot boot line.
///
/// Unset variables are on the line too, and deliberately: the reading a report
/// needs is "these five are the whole set and four of them are default", not a
/// line that goes empty and leaves a reader unsure whether it ran.
pub fn report_line() -> String {
    let mut out = String::from("vgpu_env");
    for name in ALL {
        let (state, value) = read(name);
        let short = name.strip_prefix("REIMS_VGPU_").unwrap_or(name);
        let state = match state {
            Switch::Unset => "unset".to_owned(),
            Switch::On => "on".to_owned(),
            Switch::Off => "off".to_owned(),
            // The raw value, because an operator who typed `REIMS_VGPU_GPU_STAMP=disabled`
            // needs to see what the parse rejected, not just that it did.
            Switch::Unrecognized => format!("unrecognized({})", value.unwrap_or_default()),
        };
        out.push_str(&format!(" {}={state}", short.to_ascii_lowercase()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One process-wide lock for every test that mutates the environment.
    /// `set_var` is process-global and unsynchronized; two tests setting
    /// different variables concurrently is fine, but two setting the *same* one
    /// is not, and these all touch the same probe name.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Set `PROBE` to `value` (or unset it), run `body`, and restore.
    fn with_probe<R>(value: Option<&str>, body: impl FnOnce() -> R) -> R {
        const PROBE: &str = "REIMS_VGPU_TEST_PROBE";
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the lock above serializes every mutation of this variable in
        // this process, and nothing outside these tests reads it.
        unsafe {
            match value {
                Some(v) => std::env::set_var(PROBE, v),
                None => std::env::remove_var(PROBE),
            }
        }
        let out = body();
        unsafe { std::env::remove_var(PROBE) };
        out
    }

    fn probe(value: Option<&str>) -> Switch {
        with_probe(value, || switch("REIMS_VGPU_TEST_PROBE"))
    }

    /// Both directions, in every spelling the module claims to accept. A
    /// spelling that silently reads as `Unrecognized` is a switch an operator
    /// sets and watches do nothing.
    #[test]
    fn every_documented_spelling_parses() {
        for on in ON_SPELLINGS {
            assert_eq!(probe(Some(on)), Switch::On, "{on}");
            assert_eq!(probe(Some(&on.to_ascii_uppercase())), Switch::On, "{on}");
        }
        for off in OFF_SPELLINGS {
            assert_eq!(probe(Some(off)), Switch::Off, "{off}");
            assert_eq!(probe(Some(&off.to_ascii_uppercase())), Switch::Off, "{off}");
        }
    }

    /// An unset variable and one exported empty are the same answer. `FOO=$BAR`
    /// with `BAR` unset produces the second, and reading it as a value would
    /// make an unrelated typo elsewhere in a boot script silently flip a rail.
    #[test]
    fn unset_and_empty_are_the_same_answer() {
        assert_eq!(probe(None), Switch::Unset);
        assert_eq!(probe(Some("")), Switch::Unset);
        assert_eq!(probe(Some("   ")), Switch::Unset);
    }

    /// A typo is its own answer and keeps its value, so the caller's refusal can
    /// quote what was actually written. Collapsing this into `Unset` is how a
    /// misspelled switch reads as working.
    #[test]
    fn a_value_that_is_neither_keeps_itself_for_the_message() {
        let (state, value) = with_probe(Some("mabye"), || read("REIMS_VGPU_TEST_PROBE"));
        assert_eq!(state, Switch::Unrecognized);
        assert_eq!(value.as_deref(), Some("mabye"));
    }

    /// Surrounding whitespace is not a value. A trailing space picked up from a
    /// heredoc or a `docker run -e` line would otherwise read as a typo.
    #[test]
    fn surrounding_whitespace_is_trimmed() {
        assert_eq!(probe(Some(" off ")), Switch::Off);
        assert_eq!(probe(Some("\t1\n")), Switch::On);
    }

    /// The two lists cannot overlap and are compared lowercased, so an entry
    /// with a capital in it would never match anything.
    #[test]
    fn the_spellings_are_disjoint_and_lowercase() {
        for on in ON_SPELLINGS {
            assert!(!OFF_SPELLINGS.contains(&on), "{on} is in both lists");
            assert_eq!(on, on.to_ascii_lowercase(), "{on} would never match");
        }
        for off in OFF_SPELLINGS {
            assert_eq!(off, off.to_ascii_lowercase(), "{off} would never match");
        }
    }

    /// Every variable the crate honors is named here, spelled consistently. A
    /// name that does not carry the crate prefix is one an operator cannot find
    /// by grepping their own environment.
    #[test]
    fn every_name_carries_the_crate_prefix() {
        // `ALL` rather than a second list: a list written twice is the thing
        // this module exists to stop, and the boot line reads the same one.
        let names = ALL;
        for name in names {
            assert!(name.starts_with("REIMS_VGPU_"), "{name}");
            assert!(
                name.bytes()
                    .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_'),
                "{name}"
            );
        }
        for (i, a) in names.iter().enumerate() {
            for b in &names[i + 1..] {
                assert_ne!(a, b, "two variables share a name");
            }
        }
    }

    /// The boot line names every variable, including the ones nobody set.
    ///
    /// A line that only reported what was set would go empty on a default boot,
    /// and an empty line cannot be told from an absent one — so a report from a
    /// machine with a rail switched off would look exactly like a report from a
    /// machine with a build that never emitted it.
    #[test]
    fn the_boot_line_names_every_variable_set_or_not() {
        let line = report_line();
        assert!(line.starts_with("vgpu_env "), "{line}");
        for name in ALL {
            let short = name
                .strip_prefix("REIMS_VGPU_")
                .expect("the prefix is asserted above")
                .to_ascii_lowercase();
            assert!(line.contains(&format!(" {short}=")), "{short} in {line}");
        }
    }

    /// A value the parse rejects reaches the line verbatim. An operator who
    /// wrote `disabled` instead of `off` otherwise reads `unset` and concludes
    /// the switch does not work.
    #[test]
    fn an_unrecognized_value_reaches_the_boot_line() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the lock serializes every mutation of this variable in this
        // process; `report_line` below is the only reader.
        unsafe { std::env::set_var(GUEST_IMPORT, "disabled") };
        let line = report_line();
        unsafe { std::env::remove_var(GUEST_IMPORT) };
        assert!(
            line.contains("guest_import=unrecognized(disabled)"),
            "{line}"
        );
    }
}
