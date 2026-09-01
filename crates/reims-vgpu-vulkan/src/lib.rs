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
//! - [`memory`] — how host and device memory relate on the bound physical
//!   device, and which memory type an allocation gets. A misclassification here
//!   is a performance bug and never a correctness one: topology selects a
//!   preference order, the required flags are always the fallback, and nothing
//!   may branch on topology in a way the guest can observe.
//! - [`queues`] — which queue family this rail submits to, and the value that
//!   makes a `VkQueue` have exactly one owner.
//! - [`barrier`] — what a declared barrier becomes on this host, and which of
//!   the guest's stages this host has no equivalent for.
//! - [`pools`] — one command pool per worker, and the rule that a command
//!   buffer is recordable again only when the timeline says the GPU is done
//!   with it.
//! - [`timeline`] — which value a submission signals, and the bookkeeping that
//!   keeps "reserved" and "reached" from being confused for each other.

// Calling Vulkan is unsafe by construction; a `forbid` here would be a lie the
// first entry point breaks. `unsafe_op_in_unsafe_fn` is the rule that actually
// carries: an unsafe function's body still has to say where it is relying on a
// precondition.
#![forbid(unsafe_op_in_unsafe_fn)]

pub mod barrier;
pub mod memory;
pub mod pools;
pub mod queues;
pub mod timeline;
