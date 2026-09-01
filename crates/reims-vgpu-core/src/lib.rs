//! Reims vGPU core: the semantic model, with no host in it.
//!
//! # What this crate is for
//!
//! The device's execution architecture is being replaced by one where every
//! accepted guest packet is an ordered transaction with an exact dependency
//! envelope, and where the only things that advance semantic state are
//! immutable facts returned by an executor. That model has to exist somewhere
//! that cannot reach a Vulkan handle, a Metal object, a QEMU structure or a
//! guest-RAM pointer — otherwise "backend-neutral" is a habit, and habits do
//! not survive the first convenient import.
//!
//! So it is a crate, and its dependency list is the claim: it can see what a
//! wire tag *means* and nothing about how a host produces it.
//!
//! # The parts
//!
//! - [`identity`] — the names. Guest ordinals are parsed once into total types
//!   and carried; slot reuse produces a new generation; a wrapping completion
//!   timeline is a type that knows it wraps rather than an integer every reader
//!   has to remember about.

#![forbid(unsafe_code)]

pub mod access;
pub mod bind;
pub mod blit;
pub mod compute;
pub mod content;
pub mod control;
pub mod depend;
pub mod encoder;
pub mod exec;
pub mod executor;
pub mod heap;
pub mod icb;
pub mod identity;
pub mod interpret;
pub mod lifecycle;
pub mod namespace;
pub mod operation;
pub mod pass;
pub mod pipeline;
pub mod prereq;
pub mod present;
pub mod publish;
pub mod query;
pub mod range_set;
pub mod ready;
pub mod render;
pub mod resolve;
pub mod resource_state;
pub mod retire;
pub mod schedule;
pub mod session;
/// The storage mode a resource declares, re-exported from the protocol layer.
///
/// A semantic input to placement and coherence, so the model carries it — and
/// the executor that consumes it sees this crate rather than the protocol one.
pub use reims_vgpu_protocol::storage_mode;

/// The three-dimensional extent and mip arithmetic of the guest API,
/// re-exported from the protocol layer.
pub use reims_vgpu_protocol::extent;

/// The guest's pixel-format ordinals and what each one is made of,
/// re-exported from the protocol layer.
///
/// Semantic, not native: which aspects a format has and how wide its texels
/// are decide what a transfer moves and which attachment slot an image can
/// take, on either rail. The executor that turns one of these into a native
/// format sees this crate rather than the protocol one.
pub use reims_vgpu_protocol::pixel_format;

/// What a texture declaration is, re-exported from the protocol layer.
///
/// A checked shape is a semantic input to allocation, view expansion and
/// transfer arithmetic alike, so the model carries it — and the executor that
/// turns one into a native image sees this crate rather than the protocol one.
pub use reims_vgpu_protocol::texture_shape;

pub mod stream;
pub mod submit;
pub mod sync;
pub mod transaction;
