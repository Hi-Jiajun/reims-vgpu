//! The Vulkan rail: what a host device offers, and what this device does with
//! that.
//!
//! # Why this is a crate
//!
//! The architecture names one layer as the sole executor of the semantic model
//! and the only place host capabilities become placement and transfer policy.
//! "Only place" is a claim a module boundary cannot hold — a placement decision
//! one `use` away from device state will eventually read some — so it is a
//! crate whose dependency list says what it may see: `ash`, the semantic model,
//! the refusal vocabulary, and the operator switches. No QEMU, no device model,
//! no guest-RAM ownership, no decode.
//!
//! # The rule every gate here follows
//!
//! Gate on the **capability**, never on a driver name, a vendor id, an API
//! version, or `VK_KHR_portability_subset`. Vulkan 1.2 is the baseline on every
//! supported host; anything newer needs a capability-gated fallback. An
//! operator switch may narrow what this rail does and may never widen it.
//!
//! # The parts
//!
//! - [`bindings`] — what a draw has to re-emit, and far more often what it
//!   does not.
//! - [`buffer`] — why every guest buffer here can be bound as every class,
//!   and the two things a device can still refuse about one.
//! - [`census`] — what this physical device offers, taken once, and the floor
//!   it has to clear to be used at all. Every capability gate below reads from
//!   it, and it holds no device name for one to branch on.
//! - [`device`] — the `VkDevice`, the set of features and extensions its
//!   census admitted, and the identity every handle made from it carries.
//! - [`descriptor`] — which mechanism carries a draw's descriptors on this
//!   host, and what one emission is therefore allowed to write.
//! - [`frames`] — swapchain frame slots, and the binary semaphores that a
//!   failed frame leaves with a signal outstanding on them.
//! - [`host`] — the instance, the physical device this rail bound, and the
//!   only Vulkan state the architecture allows to be process-global.
//! - [`image`] — what a checked texture declaration becomes here, and the
//!   device query that has to admit it before anything is allocated.
//! - [`layout`] — which layout each image subresource is in, and the explicit
//!   transitions and ownership moves that get it to the next one.
//! - [`memory`] — how host and device memory relate on the bound physical
//!   device, and which memory type an allocation gets. A misclassification here
//!   is a performance bug and never a correctness one: topology selects a
//!   preference order, the required flags are always the fallback, and nothing
//!   may branch on topology in a way the guest can observe.
//! - [`recording`] — everything one native recording owns, held as one value
//!   from the slots it takes to the pipelines it releases.
//! - [`queues`] — which queue family this rail submits to, and the value that
//!   makes a `VkQueue` have exactly one owner.
//! - [`barrier`] — what a declared barrier becomes on this host, and which of
//!   the guest's stages this host has no equivalent for.
//! - [`placement`] — where a resource's bytes live, decided in one order from
//!   the guest's declaration and this host's measured capabilities.
//! - [`pools`] — one command pool per worker, and the rule that a command
//!   buffer is recordable again only when the timeline says the GPU is done
//!   with it.
//! - [`variant`] — the native pipelines one semantic pipeline turns into, who
//!   is allowed to compile each, and why none of them is ever evicted.
//! - [`submission`] — what happens to a timeline point between reserving it
//!   and the GPU reaching it, and why a refused one is signalled rather than
//!   forgotten.
//! - [`timeline`] — which value a submission signals, and the bookkeeping that
//!   keeps "reserved" and "reached" from being confused for each other.

// Calling Vulkan is unsafe by construction; a `forbid` here would be a lie the
// first entry point breaks. `unsafe_op_in_unsafe_fn` is the rule that actually
// carries: an unsafe function's body still has to say where it is relying on a
// precondition.
#![forbid(unsafe_op_in_unsafe_fn)]

pub mod barrier;
pub mod bindings;
pub mod buffer;
pub mod census;
pub mod descriptor;
pub mod device;
pub mod frames;
pub mod host;
pub mod image;
pub mod layout;
pub mod memory;
pub mod placement;
pub mod pools;
pub mod queues;
pub mod recording;
pub mod submission;
pub mod timeline;
pub mod variant;
