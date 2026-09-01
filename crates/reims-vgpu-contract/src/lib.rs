//! Stable contract facts: layouts, formats, arithmetic, little-endian readers.
//!
//! Pure data and pure functions only — no guest state, no GPU, no QEMU.
//! Source of truth for numbers that come from the wire/SDK/`*_format.h`
//! contracts.
//!
//! # Backend-neutral, and now provably so
//!
//! This is the protocol vocabulary: decoded enums, descriptors, geometry, pixel
//! formats, pass actions, page arithmetic, and the exact refusals each check
//! names. What belongs *above* it is everything that knows how the host draws —
//! Vulkan handles, SPIR-V, memory placement, descriptors, queue families, image
//! layouts, host capability policy — and everything that knows the device is
//! attached to QEMU.
//!
//! That was a rule stated in this doc and held by habit while every module sat
//! in one crate. As a crate it is a fact: `ash`, Metal, QEMU and the device's
//! own state are not in scope here, so a contract check cannot reach one by
//! accident, and a reviewer does not have to notice that it did.
//!
//! # What has left for `reims-vgpu-protocol`
//!
//! This crate is being absorbed by `reims-vgpu-protocol`, which is the layer
//! the architecture names as the first one allowed to say what a wire tag
//! means. Modules move as they become movable: protocol is `no_std` and this
//! crate's refusal vocabulary is not, so the observe-free modules go first.
//! `extent` — the guest API's three-dimensional extent and its tightly-packed
//! image arithmetic — is now `reims_vgpu_protocol::extent`, and `pass_action` —
//! the `MTLLoadAction`/`MTLStoreAction` ordinals — is
//! `reims_vgpu_protocol::pass_action`. This crate uses both from there rather
//! than keeping a second spelling.
//!
//! The one dependency that looks like a device dependency and is not is
//! `reims_vgpu_observe`: a check that refuses has to be able to *name* its
//! refusal, and the [`Decline`](reims_vgpu_observe::Decline) vocabulary is that
//! name. It carries no policy and selects nothing.

pub mod draw;
pub mod gva;
pub mod gva_resolve;
pub mod iosurface_pages;
pub mod mipmap;
pub mod pixel_format;

// Moved into `reims-vgpu-protocol`, which is the layer that assigns meaning to
// a wire tag and is where this vocabulary belongs. Re-exported at the old path
// so the move is one commit and the call-site rewrite is another.
pub use reims_vgpu_protocol::{checked, dispatch, endian, fnv, vertex_step, visibility};

pub use checked::*;
