//! Reims vGPU protocol: the first layer allowed to say what a wire tag *means*.
//!
//! # Why this is a crate
//!
//! `reims-vgpu-wire` answers one question and refuses the next one. It says
//! which bytes a serializer record is made of, and it is deliberately unable to
//! say what the device owes the guest in return — a view that knew that would be
//! a view with a policy in it. Everything above wire has historically answered
//! the second question wherever it happened to be standing: a decode arm, an
//! executor match, a census route name, a comment. That is why the same
//! operation could be a no-op in one rail's reading and a dropped command in
//! another's, with nothing able to compare the two.
//!
//! This crate is the one place the second question is answered, and it depends
//! only on wire so the compiler can keep it that way: no device state, no
//! backend, no host OS, no allocation policy. `#![no_std]`.
//!
//! # The parts
//!
//! - [`closure`] — the refusal-closure ledger. For every decodable operation,
//!   exactly one outcome: implemented, contract-proven no-op on a named
//!   capability cell, contract-proven unsupported with its exact refusal, or
//!   unresolved and therefore blocking. "The current workload does not use it"
//!   and "the old backend drops it" are not outcomes and cannot be spelled here.
//! - [`packets`] — the same ledger for the FIFO packet classes, which are the
//!   other half of what a guest sends and which the manifest cannot enumerate.
//! - [`blit`] — which transfer a blit opcode names, which is its shape rather
//!   than its closure.
//! - [`extent`] — the guest API's three-dimensional extent, its mip-level
//!   dimensions, and the byte arithmetic of a tightly-packed image.
//! - [`sync`] — which opcode is a fence, an event or a barrier, on which rail,
//!   and what a barrier's scope word names.
//! - [`segment`] — what a segment-type byte means: which encoder wrote it, and
//!   which rail its records are read on.
//! - [`residency`] — what a `useResource`/`useHeap` declaration says, split so
//!   that the half a per-draw binder owes nothing on cannot hide the half it
//!   does.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_op_in_unsafe_fn)]

// `extent::tight_pyramid_spans` returns one span per mip level, a count the
// caller does not know in advance. `alloc` is still `no_std`; what this crate
// must not reach is the host, not the heap.
extern crate alloc;

pub mod blit;
pub mod closure;
pub mod extent;
pub mod packets;
pub mod residency;
pub mod segment;
pub mod sync;
