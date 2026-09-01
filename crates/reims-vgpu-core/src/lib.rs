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
pub mod depend;
pub mod encoder;
pub mod exec;
pub mod icb;
pub mod identity;
pub mod interpret;
pub mod operation;
pub mod pass;
pub mod pipeline;
pub mod prereq;
pub mod publish;
pub mod range_set;
pub mod ready;
pub mod render;
pub mod resolve;
pub mod resource_state;
pub mod retire;
pub mod schedule;
pub mod session;
pub mod stream;
pub mod sync;
pub mod transaction;
