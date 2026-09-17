//! The owner half of the canonical command channel, as the render rail uses it
//! (`research/docs/23` §86, `research/docs/26` §R9j).
//!
//! The canonical provider is driven either in-process (what the production
//! seam does today, through `metal-api-core` / `metal-api-vulkan` values) or
//! across a process boundary, where the same request travels as an `MCC1`
//! frame (`metal-api-ipc::command`, `research/docs/13`). Until this module the
//! reims device had **no producer and no consumer of that wire at all**: every
//! field a frame carries — the v83 stage-buffer pair included — reached the
//! provider only as Rust values, so "the wire does not carry it" was a true
//! statement about this device's traffic whatever the codec could encode.
//!
//! This module is the owner's own producer and consumer for the two frames
//! this rail's class answers on, and it uses the provider's decoder rather
//! than a second spelling of the format:
//!
//! - [`stage_buffer_support`] encodes the provider's own capability snapshot
//!   into a `Capabilities` response frame and reads the stage-buffer bits back
//!   out of the *decoded* value. The class gate therefore asks what the wire
//!   carries — `supports_render_stage_buffers` / `max_render_stage_buffers`,
//!   which R9h gave a tail of their own (`CAPABILITY_STAGE_BUFFER_TAIL`) —
//!   rather than what the in-process snapshot happens to hold;
//! - [`submit_frame`] / [`carried_submission`] encode the trace this rail
//!   built and decode it again with the provider's own decoder. A submission
//!   whose stage-buffer declaration the wire cannot carry is a typed decline
//!   here, at the seam, rather than a difference discovered when the owner and
//!   the provider are two processes.
//!
//! # The additive policy this module keeps
//!
//! A frame that carries no stage buffer keeps the bytes it had before the
//! declaration half existed (R9h's regression, 7/7 byte-identical frames), and
//! this rail keeps the path it had: `submit_narrow` crosses the wire only for
//! the population whose declaration is what this increment states. The grade
//! from "Rust values" to "the bytes that would travel" is a narrowing of that
//! population and not a change to any other shape's route.

use metal_api_core::provider::{
    ComputeTrace, DeviceEpoch, ProviderCapabilities, ResourceTableSnapshot, TextureFormat,
};

use metal_api_ipc::codec::CodecError;
use metal_api_ipc::command::{CommandRequest, CommandResponse};
use metal_api_ipc::command_codec::CommandCodec;

/// Why one frame could not be produced or consumed, with the step that
/// answered.
///
/// The step names the codec call rather than the shape: a frame this device
/// states and cannot carry is a fact about the wire, and the provider's own
/// decoder is the only thing that gets to say so.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WireDecline {
    pub step: &'static str,
    pub detail: String,
}

impl WireDecline {
    fn new(step: &'static str, error: CodecError) -> Self {
        Self {
            step,
            detail: error.to_string(),
        }
    }
}

/// The stage-buffer half of one provider capability snapshot, read back out of
/// the response frame the provider would send.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StageBufferSupport {
    /// Whether the provider declares it executes a pass that binds a
    /// pipeline-level stage buffer at all.
    pub supported: bool,
    /// The most stage buffer bindings one pass may carry, as the same frame
    /// declares it.
    pub maximum: u32,
}

/// Encode a capability answer, decode it again, and read the stage-buffer bits
/// out of the decoded value.
///
/// One snapshot, two readings — the provider's own and the wire's — and it is
/// the wire's that the class gate gets, because a bit the frame cannot carry is
/// a bit no remote owner would ever see.
pub fn stage_buffer_support(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<StageBufferSupport, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    Ok(StageBufferSupport {
        supported: decoded.supports_render_stage_buffers,
        maximum: decoded.max_render_stage_buffers,
    })
}

/// The compute-texture half of one provider capability snapshot, read back out
/// of the response frame the provider would send (`research/docs/26` §21.3,
/// C1c's `CAPABILITY_COMPUTE_TEXTURE_TAIL`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeTextureSupport {
    /// Whether the provider declares it samples compute-side textures at all.
    pub supported: bool,
    /// The most compute textures one pass may carry, as the same frame
    /// declares it.
    pub maximum: u32,
    /// The formats the same frame admits for a sampled compute texture.
    pub formats: Vec<TextureFormat>,
}

/// Encode a capability answer, decode it again, and read the compute-texture
/// bits out of the decoded value.
///
/// The same one-snapshot-two-readings rule [`stage_buffer_support`] states:
/// the class gate asks what the wire carries — `supports_compute_texture_sampling`,
/// `max_compute_textures` and `supported_compute_texture_formats`, which C1c
/// gave a capability section of their own — rather than what the in-process
/// snapshot happens to hold. A device whose frame cannot carry the bits is a
/// device whose remote owner never sees them, and the textured pass keeps the
/// engine under the class gate's own name for that.
pub fn compute_texture_support(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<ComputeTextureSupport, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    Ok(ComputeTextureSupport {
        supported: decoded.supports_compute_texture_sampling,
        maximum: decoded.max_compute_textures,
        formats: decoded.supported_compute_texture_formats,
    })
}

/// The provider's own capability snapshot as it comes back out of the frame
/// the owner would receive.
///
/// One encoder and one decoder for every capability question this rail asks,
/// so the readings beside each other really are one frame's bytes rather than
/// two spellings of them.
fn capabilities_frame(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<ProviderCapabilities, WireDecline> {
    let frame = CommandCodec::encode_response(&CommandResponse::Capabilities {
        epoch,
        capabilities: capabilities.clone(),
    })
    .map_err(|error| WireDecline::new("capabilities_frame", error))?;
    note_capability_frame();
    match CommandCodec::decode_response(&frame)
        .map_err(|error| WireDecline::new("capabilities_decode", error))?
    {
        CommandResponse::Capabilities { capabilities, .. } => Ok(capabilities),
        other => Err(WireDecline {
            step: "capabilities_decode",
            detail: format!("the frame decoded as a {} response", other.kind()),
        }),
    }
}

/// Encode one submission's trace and resource snapshot as the frame an owner
/// sends.
///
/// The bytes are the payload: everything this rail states about the pass — the
/// pipeline contract's stage-buffer declarations, the pass's own views, the
/// three binding sources — travels inside this frame or the encode refuses by
/// name.
pub fn submit_frame(
    trace: &ComputeTrace,
    resources: &ResourceTableSnapshot,
) -> Result<Vec<u8>, WireDecline> {
    CommandCodec::encode_request(&CommandRequest::Submit {
        trace: trace.clone(),
        resources: resources.clone(),
    })
    .inspect(|frame| note_frame(frame))
    .map_err(|error| WireDecline::new("submit_frame", error))
}

/// Decode a submission frame with the provider's own decoder, which is what a
/// provider process runs before its admission sees anything.
pub fn carried_submission(
    frame: &[u8],
) -> Result<(ComputeTrace, ResourceTableSnapshot), WireDecline> {
    match CommandCodec::decode_request(frame)
        .map_err(|error| WireDecline::new("submit_decode", error))?
    {
        CommandRequest::Submit { trace, resources } => Ok((trace, resources)),
        other => Err(WireDecline {
            step: "submit_decode",
            detail: format!("the frame decoded as a {} request", other.kind()),
        }),
    }
}

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

static CAPABILITY_FRAMES: AtomicU64 = AtomicU64::new(0);
static SUBMIT_FRAMES: AtomicU64 = AtomicU64::new(0);

/// Whether a test asked for the frames this process produces, and the frames
/// themselves while it did.
///
/// Test observation, in the shape this crate's other instruments use: with
/// nothing armed the cost is one relaxed load per frame and the frame is dropped
/// exactly as before — the payload *is* the codec's own `Vec`, so retaining it
/// would make the instrument the cost. Armed, the frame is kept so that a test
/// can decode the bytes *this rail* produced rather than a second spelling of
/// them; that difference is the whole point when the claim is "the wire carries
/// the declaration", which is a statement about bytes.
static CAPTURE_ARMED: AtomicBool = AtomicBool::new(false);
static CAPTURED_FRAMES: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// Arm or disarm the capture above, discarding whatever it held.
pub fn capture_submission_frames(armed: bool) {
    if let Ok(mut frames) = CAPTURED_FRAMES.lock() {
        frames.clear();
    }
    CAPTURE_ARMED.store(armed, Ordering::Relaxed);
}

/// The frames captured since the capture was armed, leaving it empty.
pub fn captured_submission_frames() -> Vec<Vec<u8>> {
    CAPTURED_FRAMES
        .lock()
        .map(|mut frames| std::mem::take(&mut *frames))
        .unwrap_or_default()
}

fn note_frame(frame: &[u8]) {
    if CAPTURE_ARMED.load(Ordering::Relaxed) {
        if let Ok(mut frames) = CAPTURED_FRAMES.lock() {
            frames.push(frame.to_vec());
        }
    }
}

/// How many frames this rail has produced for the canonical provider's
/// capability answer, and how many submissions it has carried across the wire.
///
/// Test observation, like the rail's other counters: what it answers is whether
/// the owner half ran at all, not what it carried.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WireCounts {
    pub capability_frames: u64,
    pub submit_frames: u64,
}

pub(crate) fn note_capability_frame() {
    CAPABILITY_FRAMES.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn note_submit_frame() {
    SUBMIT_FRAMES.fetch_add(1, Ordering::Relaxed);
}

/// The two counters above.
pub fn wire_counts() -> WireCounts {
    WireCounts {
        capability_frames: CAPABILITY_FRAMES.load(Ordering::Relaxed),
        submit_frames: SUBMIT_FRAMES.load(Ordering::Relaxed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame of the wrong kind is refused by name rather than read as a
    /// submission: the two directions of the channel share one codec, and a
    /// decoder that fell through to "whatever the bytes happened to hold" would
    /// admit another request's payload as this rail's trace.
    #[test]
    fn a_frame_that_is_not_a_submission_is_refused_by_name() {
        let frame = CommandCodec::encode_request(&CommandRequest::Capabilities)
            .expect("the probe request encodes");
        let refusal =
            carried_submission(&frame).expect_err("a capability request is not a submission");
        assert_eq!(refusal.step, "submit_decode");
        assert!(
            refusal.detail.contains("capabilities"),
            "the refusal names what the frame was instead: {refusal:?}"
        );
    }
}
