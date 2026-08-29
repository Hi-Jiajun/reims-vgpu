# AGENTS.md

Operating guide for agents working in this repository.

## Purpose and precedence

This project emulates Apple's paravirtualized GPU. An unmodified macOS guest uses its own GPU
drivers; the QEMU device and Rust backend decode the guest command stream and execute it through
Metal or Vulkan. We ship no guest driver.

Correctness comes from implementing the decoded API contract. Work proceeds in this order:

1. establish the observation and the exact build that produced it;
2. recover and state the relevant API contract;
3. identify the owning type or state machine and a contract-derived regression test;
4. implement;
5. validate on the affected pathway and rail;
6. commit only after the required validation is clean.

A request to “fix,” “boot,” “finish,” or “commit” does not skip an earlier gate. Urgency is not
authority to guess. If a contract term is unknown, recover it or emit a typed refusal; do not invent
behavior to keep moving.

Nested `AGENTS.md` files apply within their directory. Where they are stricter, they win.

## What belongs here

This file contains durable instructions. Measurements, hypotheses, one-off findings, and session
history do not belong here. Put them in:

- code documentation beside the behavior they explain, when the conclusion is durable;
- the commit body, when it describes one change and its validation;
- `kb/` or `journal/` (both gitignored), for investigation notes and working hypotheses.

Write persisted conclusions as API contracts, not as provenance stories. Do not name local
third-party binaries in code, comments, tests, documentation, commit messages, journals, or other
persisted files.

## Mandatory workflow for behavior changes and bug fixes

The workflow in this section is a gate, not advice. Follow it for protocol, lifetime, rendering,
memory, synchronization, performance, and visual changes. Mechanical refactors may omit steps that
cannot affect behavior, but must still preserve the relevant invariants and pass their verification.

### 1. Freeze and identify the observation

Before editing product code:

- Record the exact symptom without explaining it.
- Record pathway, rail, snapshot, backend, host, and workload or interaction.
- Establish the exact commit and dirty working-tree state that produced it.
- Preserve serial, fail-log, crash, screenshot, and register evidence before another boot overwrites
  or appends to it.
- Separate simultaneous symptoms. A black frame, guest panic, stalled submission stream, and host
  crash are four observations until evidence joins them.

Never classify a run as “before” or “after” from wall-clock timestamps. Prove which artifacts the
process loaded.

### 2. Create a boot identity record

Every boot used as evidence needs a manifest in `journal/` or an equivalent gitignored run
directory. Record at least:

- `git rev-parse HEAD`;
- whether the tree was dirty, a hash of the tracked diff, and the names and hashes of relevant
  untracked inputs;
- the hash of the actual QEMU executable run; when reims-vgpu is dynamically loaded, hash that
  library too, and when it is statically linked, record the linked executable plus the input
  static-library or build identity;
- pathway, rail, snapshot, backend, and relevant environment overrides;
- boot start time, process id, serial path, fail-log path, and probe/action used;
- whether the boot was baseline or candidate.

If the loaded artifact identity cannot be proven, the boot is not evidence about a code change.
Fix the harness or repeat the boot. Do not infer identity from a build finishing, a filename, a
timestamp, an open SSH port, or the appearance of a desktop.

Ensure the prior VM is gone before starting another. Clear `/tmp/reims-vgpu-fail.log` before each
boot; it appends across processes. Wait for the new device log as well as the guest before driving a
probe, so an old VM cannot answer SSH for a failed new boot.

### 3. Establish a real baseline

For a visual, protocol, performance, synchronization, or lifetime regression, build and run the
parent or other agreed control before the candidate. Use the same pathway, rail, snapshot, backend,
environment, and interaction.

Build controls in an isolated copy without changing this shared checkout. Do not use `checkout`,
`switch`, `stash`, `reset`, or `restore` to manufacture a baseline.

If the baseline cannot be run or the defect cannot be reproduced, say so. Instrument the defect
class and keep claims narrow; do not replace a missing baseline with an explanation.

### 4. Make the contract checkpoint

Before editing product code, write a short checkpoint in the user-visible progress update and in a
gitignored investigation note. It must contain:

- **Observed:** facts directly measured.
- **Contract:** behavior established by decoded fields, headers, layouts, serializer behavior,
  calling conventions, or host capabilities.
- **Unknown:** every decision-affecting term not yet established.
- **Owner:** the type, resolver, or state machine that must enforce the invariant.
- **Test:** the externally visible relation that will fail without the repair.

Use a decision ledger when more than one behavior is involved:

| Proposed decision | Contract term that requires it | Status |
|---|---|---|
| Example: retain a resource until command completion | command resource lifetime | established |
| Example: defer a Store until a later reader | none | prohibited |

Every guest-visible branch needs an established row. “Existing code does this,” “the pixels look
right,” “tests pass,” “it is on the same queue,” and “this mechanism is already available” are not
contract terms.

End the checkpoint with `CONTRACT GATE: PASS` only when every decision-affecting row is established.
Otherwise write `CONTRACT GATE: BLOCKED`, continue contract recovery or implement a typed refusal,
and do not edit guest-visible behavior.

### 5. Recover missing contract terms before design

Static reverse engineering is a legitimate and expected way to recover an unclear interface.
Locally available headers, symbols, strings, disassembly, call graphs, struct layouts, field access,
and calling conventions may be inspected. Controlled runtime instrumentation is also allowed when
it observes without changing guest-visible decisions.

Third-party bytes stay local:

- Never commit binaries, extracted shaders or firmware, disassembly listings, or copied excerpts.
- Never name the local binaries in anything persisted by the repository or investigation.
- Persist only the API conclusion: field meaning, opcode, offset, size, lifetime, ordering rule, or
  calling convention.

When the contract remains unknown, implement a typed refusal on the always-on fail channel. Do not
infer the answer from timing, arrival order, object id, name, allocation size, address range, pixel
content, frame count, or any other correlate.

### 6. Trace asynchronous contracts end to end

Queue submission is not guest completion. For commands involving GPU work, guest memory, resource
reuse, interrupts, or fences, trace the whole lifetime before choosing a design:

```text
decode -> retained inputs -> host submission -> host completion -> guest-visible Store/write
       -> completion word/interrupt -> guest release or reuse
```

State explicitly:

- what retains every resource;
- when host work may still read or write it;
- what makes results visible to the guest;
- what orders the completion word and interrupt after those results;
- when the guest may release or repurpose the resource or its pages;
- which owning state transition makes late work unrepresentable.

An implementation is not structural if these obligations are assembled opportunistically at a
call site. Put them in the command transaction, lifetime type, resolver, or state machine that owns
them.

### 7. Write the contract-derived test first

The regression test must express the external invariant, not the proposed mechanism. It must fail
without the change or be accompanied by a demonstrated proxy that does.

For asynchronous lifetime work, exercise release or reuse at the completion boundary and assert
that no read, write, debt, or callback can escape past it. A test that only checks the final pixels
or a later cache hit is insufficient.

Do not validate by reading this repository's Rust source as text. No source regexes, call-shape
greps, file censuses, or verdict tables keyed by file and line. Prefer, in order:

1. make invalid state unrepresentable in a type;
2. derive one value from the other rather than duplicating it;
3. use an inline `const` assertion for relations between independently derived values;
4. write a behavioral test.

The one allowed source comparison is the C ABI header against Rust constants through
`qemu::abi::header_define`, because neither language can include the other representation.
Coverage instruments such as `scripts/runtime-dead` are also allowed; they measure execution.

### 8. Implement only the established contract

Every branch must be justified by a decoded guest field, header constant, `sizeof`/`offsetof`,
serializer output, calling convention, or measured host capability.

- Parse arbitrary guest ordinals once into total Rust types; carry the type, not its integer.
- Never transmute a guest ordinal into a Metal enum. Use `backend::metal::mtl_enum` and refuse
  unknown or hole values.
- Keep guest page geometry explicit through `page_shift` or `page_size`.
- Derive constants; do not choose numbers because they fit one observation.
- Gate on host capabilities, never vendor, driver, or device names.
- Environment overrides may narrow capability, never widen it.
- Make dropped, rejected, degraded, unsupported, or mis-executed guest work visible through typed
  reasons on `/tmp/reims-vgpu-fail.log`. Expected “not ready” control flow stays quiet.
- Remove probe-only behavior before validation.

New behavior belongs in the existing owner. If the owning type cannot express it, change the type.
Do not add a flag around a resolver, a second lookup after it, a fixup pass, or a special branch in
front of the general path.

### 9. Validate in layers

Run, in order:

1. the focused regression test;
2. nearby subsystem tests;
3. the full affected backend suite, serially;
4. affected clippy and feature-matrix arms;
5. the baseline and candidate VM boots on the exact pathway and rail;
6. the original interaction plus a release/reuse stress case when lifetime is involved.

Synthetic tests do not replace a live boot for a defect that appears only in an unmodified guest.
A screenshot does not replace a behavioral or log-level gate.

### 10. Catastrophic failures invalidate the candidate

Any of the following makes a candidate ineligible to commit:

- guest kernel panic or memory-corruption signature;
- WindowServer or relevant guest-process crash;
- prolonged loss of guest submissions or fresh frames;
- fence timeout, device loss, or unexplained drain-log stop;
- host assertion, signal, crash, or sanitizer finding;
- late guest-memory write after completion or resource reuse.

Stop changing the candidate's guest-visible behavior. Preserve the exact artifacts and identity
record. Read-only diagnostics and observation-only instrumentation may continue. Report that the
candidate was active and that causality is unknown until proved; never relabel the run as pre-change
or unrelated from timing or intuition.

Do not layer a second repair onto the failed design until the failure mechanism is understood or a
properly identified control demonstrates it is independent. A green unit suite does not override a
catastrophic VM result.

### 11. Repeat before claiming success

Never report a fix from the first green boot. Repeat the passing candidate at least three times for
freeze, panic, or intermittent visual failures, and check a second relevant rail when the contract
is shared across guest versions. A single catastrophic red candidate blocks its commit until the
failure is understood; it does not, by itself, prove causation. A single green candidate does not
establish a rate.

State exactly what was verified. One workload, host, pathway, and rail proves only that combination.

## Architecture and ownership

### Main components

- `vendor/qemu`: thin QEMU device shim—QOM, MMIO/BAR, IRQ/MSI, console/display integration, and
  HostOps plumbing.
- `crates/reims-vgpu`: Rust device model, protocol decode, mapping, scheduling, command planning and
  execution, presentation, and Metal/Vulkan policy.
- `crates/reims-vgpu/src/observe/`: typed failure and census emission.
- `crates/reims-vgpu-wire`: derived wire views; its own `AGENTS.md` also applies.
- `vm/`: snapshot-revert boot scripts and rail selection.
- `bugs/`: gitignored handoff packages for translator defects.

### C is a thin shim

Product logic belongs in Rust. C and Objective-C connect QEMU to Rust. A shim must not reconstruct a
rule by combining several Rust queries; export the final answer from Rust.

Once a `reims_vgpu_qemu_*` entry point has a wrapper in `reims-vgpu-shim.c`, that wrapper is its only
caller. Check both device shims manually when changing the shared boundary because they compile on
different hosts.

Anything crossing the C/Rust boundary exists twice. Add a `qemu::abi::header_define` test for every
shared constant in `crates/reims-vgpu/include/reims_vgpu_qemu_abi.h`.

### Resource state follows guest lifetimes

A cache containing state whose loss costs guest work must be unbounded by arbitrary numbers and
keyed to a contract-owned guest lifetime. Release entries when the guest releases their owning
resource or mapping. Do not add LRU, insertion-order eviction, rotating slots, sampling strides, or
bitmasks standing in for unbounded sets.

A bound is acceptable only when eviction loses a purely derived value that can be recomputed, or
when the contract itself supplies the bound. If excess work must be refused, use a typed refusal;
never silently evict guest state.

### Broad sweeps require behavioral evidence

Before deleting or consolidating code:

- Name the guest action that reaches a branch. A decoded but unobserved arm is usually contract
  fidelity, not dead code.
- Locate where a counter is sampled. Zero can mean idle workload or wrong sampling point.
- Measure requested reach as well as drops before changing a bound.
- Compare every execution arm consuming the same wire form; do not read one in isolation.
- Treat one-slot `Option` accumulators, fixed-width masks, truncated runs, and integerized enums as
  bounds even when no capacity constant exists.

Read module documentation before sweeping. Durable findings belong beside the owner, not in this
file.

## Supported pathways

All three pathways are first-class:

| Pathway | Host | Guest | Attach | Page shift | Backend | Boot |
|---|---|---|---|---|---|---|
| x86 macOS / Linux Vulkan | Linux x86_64 KVM | x86_64 macOS | PCI (`reims-vgpu-pci`) | 12 | Vulkan | `vm/boot-x86.sh` |
| arm64 macOS / macOS Metal | Apple Silicon HVF | arm64 macOS | sysbus (`reims-vgpu-mmio`) | 14 | Metal-direct | `vm/boot-arm64.sh` |
| arm64 macOS / macOS Vulkan | Apple Silicon HVF | arm64 macOS | sysbus (`reims-vgpu-mmio`) | 14 | MoltenVK | `vm/boot-arm64.sh` |

Do not generalize between architectures, backends, host GPU classes, or guest rails. Verify the
pathway being changed.

Vulkan 1.2 is the baseline. Newer functionality needs a capability-gated fallback. The Vulkan
backend must remain correct in all four memory cells:

| | Host-pointer import | No host-pointer import |
|---|---|---|
| Unified memory | direct host-backed resources | copying path |
| Discrete memory | imported backing plus device-local copy | staging path |

`caps::memory_topology` derives topology from structural capabilities. Topology may choose a
performance path but must not change guest-visible semantics.

Guest RAM imports are sized to a RAMBlock. `runtime/guest_ram.rs` owns the bounds and provenance of
`GuestRamImport` and `GuestSlice`; add operations there instead of exposing raw pointers or offsets.
Host-pointer import is optional. Changes to guest-memory upload, bind, or writeback must also be
validated with `REIMS_VGPU_GUEST_IMPORT=off` and pixel output compared between paths.

Read environment variables only through `crates/reims-vgpu/src/env.rs`. An override is a refusal
gate, not a capability grant.

## Translator defects

Check the pinned `metal2vulkan` revision against its upstream before diagnosing a translator defect.
When the defect is upstream, create `bugs/<defect-name>/` containing:

| File | Contents |
|---|---|
| `README.md` | guest-visible loss, translator defect, location, and ruled-out causes |
| `input-*.air` | one reproducing AIR input per distinct blob |
| `failure.txt` | verbatim validator output and per-tier retry trace |
| `repro.sh` | runs every input and prints the verdict |

Name the directory after the defect, not the symptom. `bugs/` is gitignored because the payloads
are third-party bytes; hand it over by copying, never commit it.

## VM verification

Select a rail explicitly with `--rail NAME`; `--list-rails` lists them. Name the rail in every
reported result.

- x86: `vm/boot-x86.sh --device reims-vgpu-pci --testing --rail NAME`
- arm64: `vm/boot-arm64.sh --device reims-vgpu-mmio --testing --rail NAME`

Use host-driven QMP input through `scripts/qmp/qmp.py` for pointer and keyboard actions. Use the host
screenshot helpers, not QMP `screendump`; the host-owned window does not necessarily pass its frame
through QEMU's display surface.

For x86 probes, run `vm/guest-authorize.sh` after the boot and before SSH-dependent probes. Bound
every guest command with a host-side timeout. SSH availability does not prove the desktop is ready;
wait for the relevant desktop process and then allow it to settle.

If the login window appears unexpectedly, preserve and collect guest diagnostic reports before
logging in. The report is the crash evidence; console ownership is not. A probe exiting zero does
not prove the boot remained healthy—inspect the boot script's own result and serial log afterward.

For performance work:

- drive a sustained workload; an idle or bursty boot measures a different population;
- run with no builds, subagents, or second VM competing for the host;
- join census records by their `t=` field, never by line ordinal;
- quote both offered and presented frames;
- report CPU and GPU cost, draw rate, and draws per frame together;
- interleave baseline and candidate runs, and use several runs per arm.

### Reading `/tmp/reims-vgpu-fail.log`

- One clean log per boot. Count `vk_caps` records to detect accidental concatenation.
- `OFF` records are observations, not failures. Filter them out before ranking fail reasons.
- Emitters deduplicate; counters do not. Never quote one as the other.
- Decoder success is generally silent. Absence of a failure line does not prove a decoder ran.
- `store_routes` values are per-window and must be summed; high-water fields such as `peak` use the
  final or maximum sample instead.
- A log that stops differs from a log that reports a failure. Compare drain, display, and host-loop
  emitters, then attach a debugger when the owning thread stops without a typed reason.

## Tests and build verification

Run Rust tests serially; GPU-touching tests are not safe in parallel.

```sh
cargo test -p reims-vgpu --no-default-features --features backend-vulkan,host-window -- --test-threads=1
```

On an Apple host, also run:

```sh
cargo test -p reims-vgpu --no-default-features --features backend-metal -- --test-threads=1
```

Metal tests are cfg-disabled on non-Apple hosts. Cross-clippy compiles that code but does not run
its tests; never report them as passed from Linux. Layout-truth tests under `reims-vgpu-wire` are
ignored without the gitignored captured fixtures. Their absence means wire layout was not verified.

Run the feature matrix when cfgs, shared Rust, feature boundaries, or backend boundaries change:

```sh
scripts/feature-matrix/feature-matrix.sh
```

The feature matrix runs `cargo check`; it does not replace clippy. Use this required arm matrix:

| Change scope | Required clippy arms |
|---|---|
| shared Rust, runtime, model, contract, decode, QEMU ABI, or public engine types | all three main arms below |
| Vulkan-only implementation with no shared signature or cfg change | both Vulkan arms below |
| Metal-only implementation with no shared signature or cfg change | the aarch64 Metal arm below |
| feature, cfg, backend boundary, or uncertain scope | all three main arms below, plus the feature matrix |
| `crates/reims-vgpu-efi` | both EFI arms below, from that workspace |

When in doubt, run all three main arms. All three can run on Linux: the Metal command
cross-compiles for `aarch64-apple-darwin`, but it does not run Metal tests. Every clippy invocation
must use `-D warnings`; zero warnings are required. Do not add an `allow`, hide a warning, weaken a
target, or skip an arm merely to make the gate pass.

The three main clippy arms are:

```sh
cargo clippy -p reims-vgpu --target aarch64-apple-darwin --all-targets \
  --no-default-features --features backend-metal -- -D warnings
cargo clippy -p reims-vgpu --all-targets \
  --no-default-features --features backend-vulkan,host-window -- -D warnings
cargo clippy -p reims-vgpu --target x86_64-unknown-linux-gnu --all-targets \
  --no-default-features --features backend-vulkan,host-window -- -D warnings
```

For `crates/reims-vgpu-efi`, run from that crate:

```sh
cargo clippy --target x86_64-unknown-uefi -- -D warnings
cargo clippy --profile test --lib -- -D warnings
```

Use the documentation link pass when moving or deleting documented APIs:

```sh
RUSTDOCFLAGS="-A rustdoc::private_intra_doc_links" cargo doc -p reims-vgpu \
  --no-deps --document-private-items \
  --no-default-features --features backend-vulkan,host-window
```

Formatting is a required Rust gate. Running only `--check` does not satisfy it: run `cargo fmt`
itself first, then verify the result. For changes in the main workspace, run:

```sh
cargo fmt --all
cargo fmt --all -- --check
```

Run the same commands from a nested workspace, including `crates/reims-vgpu-efi`, when it is
changed. Review the formatting diff before committing so unrelated user edits are not accidentally
absorbed, but do not skip formatting because it changes lines touched by the task.

## Git and shared-worktree safety

Existing dirty changes belong to the user unless authorship is known. Preserve them and avoid
unrelated edits.

Never use `git checkout`, `switch`, `stash`, `reset`, or `restore` to prepare a test, baseline, or
revert a probe. Edit a temporary probe out the same way it was edited in, or work in an isolated
copy. Never use destructive recursive commands against broad or unresolved paths.

Subagents share this checkout. If delegation is explicitly requested, brief every subagent as
read-only with respect to git: no checkout, switch, stash, reset, restore, or commit. Check
`git status` after they finish.

## Commit requirements

Commit only work authored for the current task. Never commit third-party software, firmware, disk
images, `.mtlb`, AIR, SPIR-V, extracted assets, disassembly, or copied binary excerpts. Do not name
the local third-party binaries used for analysis in the commit message.

Do not commit a behavior change unless:

- its contract checkpoint is complete with no unsupported decisions;
- a focused contract-derived regression test fails without it and passes with it;
- `cargo fmt` has been run for every affected Rust workspace and its `--check` passes;
- affected tests and clippy arms pass;
- the affected live pathway and rail pass when the defect requires a VM;
- no candidate boot produced an unexplained catastrophic failure;
- the working tree contains no accidental or unrelated changes.

The commit body must state:

- affected component, pathway, and rail;
- the API contract implemented and the owning invariant;
- behavior changed and why;
- focused test and broader verification actually run;
- live baseline/candidate results and boot identities when applicable;
- everything not verified.

Never overstate a result. “Clippy clean” means only the listed arms. “Booted” does not mean the
interaction passed. “No failure line” does not mean the path executed. “One clean boot” does not
mean an intermittent defect is fixed.
