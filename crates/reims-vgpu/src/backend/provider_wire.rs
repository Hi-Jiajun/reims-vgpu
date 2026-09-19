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
    /// The most stage buffer bindings **one stage** of a pass may carry, as
    /// the same frame declares it; `0` is the absent section, which is the
    /// reading every frame written before E-SB2 gives (`research/docs/23`
    /// §117).
    ///
    /// The two numbers are not a list and a sub-list of one rule: `maximum` is
    /// the *pipeline-level* bound the wire's own declaration block is sized
    /// by, while this one is the ceiling the canonical contract states on a
    /// stage's own axis. A frame that carries no section states no per-stage
    /// ceiling, and the class gate then keeps the stricter merged reading
    /// rather than half-admitting the pair.
    pub per_stage: u32,
    /// Whether the same frame declares the stage-buffer **whole-binding** arm:
    /// a declaration whose reach the translation could not state, executed
    /// against the bind's own window (`research/docs/23` §3.3, E-SB3).
    ///
    /// `false` is the absent section, which is the reading every frame written
    /// before E-SB3 gives, and the fail-closed one: the walk then keeps its own
    /// by-name refusal for the shape rather than handing the provider a
    /// declaration nothing measured.
    pub binding_range: bool,
}

impl StageBufferSupport {
    /// Whether the frame states a ceiling on one stage's own list, mirroring
    /// `ProviderCapabilities::declares_render_stage_buffer_per_stage_ceiling`
    /// on the decoded side.
    ///
    /// The decoded number is what the class gate has to weigh, because a bit
    /// the frame cannot carry is a bit no remote owner would ever see: an
    /// owner-side snapshot that declared the window while the wire dropped it
    /// is exactly the failure [`capabilities_frame`] exists to catch, and the
    /// missing section is not an error but the older, stricter reading.
    pub fn declares_per_stage_ceiling(&self) -> bool {
        self.per_stage != 0
    }
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
        per_stage: decoded.max_render_stage_buffers_per_stage,
        binding_range: decoded.supports_render_stage_buffer_binding_range,
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

/// The render-texture half of one provider capability snapshot, read back out
/// of the response frame the provider would send (`research/docs/23` §3.3,
/// v70's `CAPABILITY_RENDER_TEXTURE_TAIL`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderTextureSupport {
    /// Whether the provider declares it samples render-pass textures at all.
    pub supported: bool,
    /// The most sampled textures one render *pass* may bind, as the same frame
    /// declares it — the pass's list bound (`research/docs/23` §3.3, E-TC1).
    pub maximum: u32,
    /// The most sampled textures **one stage** of a render pass may declare, or
    /// `0` for the frame that carries no per-stage section (`research/docs/23`
    /// §3.3, E-TC1).
    ///
    /// The two numbers are not a list and a sub-list of one rule: `maximum` is
    /// the pass's own list bound — the pair's sum on the canonical side — while
    /// this one is the ceiling the device states on one stage's own axis. A
    /// frame that carries no section states no per-stage window, and the class
    /// gate then keeps the pre-increment reading rather than half-admitting the
    /// wider list.
    pub per_stage: u32,
    /// The formats the same frame admits for a sampled render texture.
    pub formats: Vec<TextureFormat>,
}

impl RenderTextureSupport {
    /// Whether the frame states a ceiling on one stage's own texture list,
    /// mirroring `ProviderCapabilities::declares_render_texture_per_stage_ceiling`
    /// on the decoded side (`research/docs/23` §3.3, E-TC1).
    ///
    /// The decoded number is what the class gate has to weigh, because a window
    /// the frame cannot carry is a window no remote owner would ever see: an
    /// owner-side snapshot that declared it while the wire dropped it is
    /// exactly the failure [`capabilities_frame`] exists to catch, and the
    /// missing section is not an error but the older, stricter reading.
    pub fn declares_per_stage_ceiling(&self) -> bool {
        self.per_stage != 0
    }
}

/// Encode a capability answer, decode it again, and read the render-texture
/// bits out of the decoded value.
///
/// The third reading of the same one-snapshot rule ([`stage_buffer_support`],
/// [`compute_texture_support`]), and the one R28 needs: a sampled pass whose
/// bind is an owner lease crosses the submission frame, so the *frame's* own
/// answer — `supports_render_texture_sampling`, `max_render_textures` and
/// `supported_render_texture_formats`, the section v70 gave the tail — is what
/// decides whether the pass may leave for the provider. Asking the in-process
/// snapshot instead would read a bit no remote owner ever sees.
///
/// The section answers the *provider's* side of the question (it will create
/// the views and samplers this shape needs). That the frame also *carries* the
/// pass's own texture list and its render contract's declarations is the
/// codec's side, and it is not assumed here: it is read back from the frame
/// this rail produces before every lease-carrying submission
/// ([`carried_submission`], `render_provider_wire render_texture` lines), and
/// the rail's own tests decode a captured frame and assert the declarations and
/// the lease sources survived the round trip.
pub fn render_texture_support(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<RenderTextureSupport, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    Ok(RenderTextureSupport {
        supported: decoded.supports_render_texture_sampling,
        maximum: decoded.max_render_textures,
        per_stage: decoded.max_render_textures_per_stage,
        formats: decoded.supported_render_texture_formats,
    })
}

/// The two narrow sampled lanes of one provider capability snapshot, read back
/// out of the response frame the provider would send (`research/docs/23` §3.3,
/// E's `render-sampled-narrow-lanes` / R39).
///
/// The seventh reading of the same one-snapshot rule ([`stage_buffer_support`],
/// [`compute_texture_support`], [`render_texture_support`],
/// [`render_sampler_carriage`], [`stage_buffer_namespace_split`],
/// [`render_texture_gathered_extent`]), and it is R28's reading looked at one
/// lane at a time: the section's own format list is what decides which texels
/// the provider creates a sampled view for (`supported_render_texture_formats`,
/// the field E's narrow-lane change widened), so the two lanes this rail asks
/// about are answered by membership in *that* list and not by a second table
/// here.
///
/// `false` is the fail-closed answer for each lane, and it is what a frame
/// written before the lane existed decodes to: the two codes E appended
/// (`5`/`6`) are absent from an older frame's list, and a decoder that met one
/// is refusing the frame outright rather than reading it as another format, so
/// a device whose section lists neither lane is a device that keeps those binds
/// on the engine under the class's own name.
pub fn render_texture_narrow_lanes(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<NarrowLaneSupport, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    let formats = decoded.supported_render_texture_formats;
    Ok(NarrowLaneSupport {
        r8: formats.contains(&TextureFormat::R8Unorm),
        rg8: formats.contains(&TextureFormat::R8G8Unorm),
    })
}

/// The two narrow sampled lanes of one provider capability snapshot, one bit
/// each, as [`render_texture_narrow_lanes`] reads them out of the frame.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NarrowLaneSupport {
    /// Whether the frame's format list carries `r8_unorm`.
    pub r8: bool,
    /// Whether the same list carries `r8g8_unorm`.
    pub rg8: bool,
}

/// The eight-byte half-float sampled lane of the same section
/// (`research/docs/23` §3.3, §107; census v44's `texture_bind` bucket).
///
/// The eighth reading of the same one-snapshot rule and
/// [`render_texture_narrow_lanes`]'s sibling one width over: the section's own
/// format list is what decides which texels the provider uploads and samples
/// (`supported_render_texture_formats`, the field the 2026-09-19 widening
/// appended `rgba16_float` to), so this lane is answered by membership in
/// *that* list and not by a second table here.
///
/// `false` is the fail-closed answer, and it is what a frame written before
/// the lane existed decodes to: the code `4` is absent from an older frame's
/// list — the census's own boots all carry it as an *unlisted* format, which is
/// exactly the refusal the class states — so a device whose section does not
/// list the lane keeps those binds on the engine under the class's own name.
pub fn render_texture_rgba16f_lane(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<bool, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    Ok(decoded
        .supported_render_texture_formats
        .contains(&TextureFormat::Rgba16Float))
}

/// The one-dimensional sampled window and the two lanes it carries, read out of
/// the same capability frame (`research/docs/23` §119, census b10's
/// `texture_shape` bucket).
///
/// The twelfth reading of the same one-snapshot rule ([`stage_buffer_support`],
/// [`compute_texture_support`], [`render_texture_support`],
/// [`render_texture_narrow_lanes`], [`render_texture_rgba16f_lane`], …), and the
/// one the one-dimensional arm needs: the frame states three facts about it —
/// whether its section's format list carries `r32_float`, whether it carries
/// `r16_float`, and how wide a single-row LUT the snapshot admits
/// (`max_render_texture_dimension_1d`, the tail's own `0x00 0x0A` section) — and
/// they are one answer rather than three questions: a lane without a window
/// names no width, and a window without a lane names no texels.
///
/// `false`/`0` is the fail-closed reading of each half, and it is what a frame
/// written before the arm existed decodes to: the two format codes are absent
/// from an older list (and a decoder that met one refuses the frame by unknown
/// code rather than reading another format), and the section itself is absent
/// from an older tail — the tag is the escape family's next one, so a decoder
/// that predates it answers [`WireDecline`] rather than a value, and a decoder
/// that carries the tag reads `0` out of a frame that ends before it. A device
/// that states none of the three is a device that keeps its own refusal by name
/// for the shape.
pub fn render_texture_one_dimension_window(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<OneDimensionSupport, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    Ok(OneDimensionSupport {
        r32f: decoded
            .supported_render_texture_formats
            .contains(&TextureFormat::R32Float),
        r16f: decoded
            .supported_render_texture_formats
            .contains(&TextureFormat::R16Float),
        window: decoded.max_render_texture_dimension_1d,
    })
}

/// Whether the provider whose snapshot this is executes a *layout-free*
/// non-indexed draw whose count is above the milestone's three vertices
/// (2026-09-19, census v45's `vertex_span` bucket).
///
/// The fourteenth reading of the same one-snapshot rule
/// ([`render_texture_one_dimension_window`] is the thirteenth), and the one
/// R39's vertex-span gate needs now that the canonical contract admits the
/// count from three vertices up: the shape is well formed whoever executes it,
/// so the only thing left to ask is whether *this* provider runs it. The
/// contract's own widening is what makes the two questions separate — before
/// it, a count above three was refused by name on every rail, and the class's
/// refusal was the only answer.
///
/// `false` is the fail-closed reading, and it is what a frame written before
/// the arm existed decodes to: the bit is the escape family's next tagged
/// section (`0x00 0x0B`), so a decoder that predates the tag answers
/// [`WireDecline`] rather than a value, and one that carries the tag reads
/// `false` out of a frame that ends before it. A device that does not state the
/// bit is a device that keeps the census's refusal, its slug and its sentence,
/// for the shape — which is exactly the native rail's answer, whose reviewed
/// `vertex_id` module carries three positions.
pub fn render_vertex_count_above_triangle(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<bool, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    Ok(decoded.supports_render_vertex_count_above_triangle)
}

/// Whether the provider whose snapshot this is executes a fragment module that
/// stores **more colour locations than the pass attaches** (2026-09-20, the
/// third door behind census v46's `stage_buffer_footprint` bucket).
///
/// The fifteenth reading of the same one-snapshot rule
/// ([`render_vertex_count_above_triangle`] is the fourteenth), and the one the
/// class gate needs for the shape census v46's remaining LPF pipeline has: its
/// fragment stage stores three colour locations while the draw attaches one.
/// Vulkan defines what the provider does with the extra stores — a fragment
/// output whose location has no attachment behind it is discarded — so the
/// shape is executable; the only thing left to ask is whether *this* provider's
/// registration gate admits it. A provider that does not declare the face
/// refuses the registration by name (`render_stage_reflection_mismatch`), and a
/// draw the class handed such a provider is a draw no rail answered — the census
/// red line `draws_skipped_after_engine_refusal`.
///
/// `false` is the fail-closed reading, and it is what a frame written before
/// the face existed decodes to: the bit is the escape family's own next tagged
/// section (`0x00 0x0E`), so a decoder that predates the tag answers
/// [`WireDecline`] rather than a value, and one that carries the tag reads
/// `false` out of a frame that ends before it. A device that does not state the
/// bit is a device that keeps the census's slug and sentence for the shape —
/// which is exactly the native rail's answer, whose reviewed-module table
/// selects a stage by the colour format list's exact shape.
pub fn render_fragment_output_superset(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<bool, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    Ok(decoded.supports_render_fragment_output_superset)
}

/// The three facts one provider's frame states about its one-dimensional
/// sampled window, one value each.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OneDimensionSupport {
    /// Whether the frame's format list carries `r32_float`.
    pub r32f: bool,
    /// Whether the same list carries `r16_float`.
    pub r16f: bool,
    /// The widest single-row LUT the frame admits, in texels of that row; `0`
    /// for a frame that does not carry the section at all.
    pub window: u64,
}

/// The three-dimensional sampled arm's lanes and window, read out of the same
/// capability frame (2026-09-20, the `D3` sampled texture arm and, one
/// increment later, census v48's volume lane gate).
///
/// [`render_texture_one_dimension_window`]'s sibling two axes over, and the
/// fifteenth reading of the same one-snapshot rule: the frame states two facts
/// about the arm — **which lanes** it admits for a volume
/// (`supported_render_texture_volume_formats`, the tail's own `0x00 0x11`
/// section) and how large a volume it admits per axis
/// (`max_render_texture_dimension_3d`, the `0x00 0x0D` section beside it) — and
/// they are one answer rather than two questions, because a lane without a
/// window names no volume and a window without a lane names no texels.
///
/// The lane half is a *device* answer and not a reading of the render-sampler
/// block's format list: a device may list `bgra8_unorm` as a sampled surface
/// and still refuse a `TYPE_3D` image in that lane, which is exactly the
/// reading the `D3` arm's own first increment met (the RTX 5060 refuses the
/// linear `R32_SFLOAT` volume the pre-device-copy arm created while Lavapipe
/// accepts it). The census's remaining LPF pipeline binds its three volumes in
/// the `B8G8R8A8_UNORM` lane, so the list is what decides whether that bind
/// travels.
///
/// The window is **per axis**, not a texel count: Vulkan's
/// `maxImageDimension3D` bounds a `TYPE_3D` image's width, height *and* depth,
/// so a caller holds each of the bind's three extents to it rather than their
/// product — which is the one way this reading differs from the
/// one-dimensional window's own.
///
/// The absent sections are the older readings, and each one is the rule the arm
/// carried before its own increment: a frame written before the *lane* section
/// existed reads as `r32_float` alone — the one lane the `D3` arm shipped with
/// — and a frame that ends before the window reads `0`, the fail-closed
/// direction. That is why an old frame's meaning does not move: every lane
/// beside `r32_float` keeps its refusal by name under the older reading, which
/// is what the census counted before this increment.
pub fn render_texture_dimension_3d_window(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<VolumeSupport, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    // The lane list is the device's own answer, and its **absence** is the
    // pre-increment reading *verbatim* rather than an empty set or an
    // unconditional `r32_float`: before the section existed the arm's lane came
    // out of the render-sampler block's own format list, so a frame written
    // before it keeps exactly that rule — `r32_float` where the surface list
    // carries it, and no lane where it does not. A provider that lists lanes
    // states the closed set instead, and a frame that names none of them is one
    // whose device has no volume lane at all.
    let lanes = if decoded.declares_render_texture_volume_formats() {
        VolumeLanes::of(&decoded.supported_render_texture_volume_formats)
    } else if decoded
        .supported_render_texture_formats
        .contains(&TextureFormat::R32Float)
    {
        VolumeLanes::of(&[TextureFormat::R32Float])
    } else {
        VolumeLanes::NONE
    };
    Ok(VolumeSupport {
        lanes,
        window: decoded.max_render_texture_dimension_3d,
    })
}

/// The lanes one provider's frame lists for its three-dimensional sampled arm
/// (2026-09-20, census v48's volume lane gate).
///
/// One bit per contract lane rather than a `Vec<TextureFormat>` for the two
/// reasons the window beside it is a number: the whole answer travels as one
/// `Copy` value through the pure class gate, and the family is closed — the
/// contract names every format a `TYPE_3D` sampled image can carry on this rail,
/// and the wire section is decoded against the same closed list.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VolumeLanes(u8);

impl VolumeLanes {
    /// The set a frame that lists no lane at all states.
    pub const NONE: Self = Self(0);

    /// The bit one contract format occupies in the set, or `None` for a format
    /// this reading cannot state as a volume lane.
    ///
    /// The seven are [`TextureFormat::RENDER_SAMPLED`]'s own: the lanes the
    /// render sampler admits as a sampled source are the only ones the rail's
    /// volume arm can name, so a frame that listed another format would name a
    /// lane no declaration of this class could restate.
    const fn bit(format: TextureFormat) -> Option<u8> {
        match format {
            TextureFormat::Rgba8Unorm => Some(1 << 0),
            TextureFormat::Bgra8Unorm => Some(1 << 1),
            TextureFormat::R8Unorm => Some(1 << 2),
            TextureFormat::R8G8Unorm => Some(1 << 3),
            TextureFormat::Rgba16Float => Some(1 << 4),
            TextureFormat::R32Float => Some(1 << 5),
            TextureFormat::R16Float => Some(1 << 6),
            TextureFormat::R32Uint => None,
        }
    }

    /// The set one frame's decoded lane list states.
    pub fn of(formats: &[TextureFormat]) -> Self {
        formats
            .iter()
            .fold(Self::NONE, |set, format| set.with(*format))
    }

    /// The same set with one more lane in it; a format this reading cannot
    /// state leaves the set alone rather than inventing a bit for it.
    pub const fn with(self, format: TextureFormat) -> Self {
        match Self::bit(format) {
            Some(bit) => Self(self.0 | bit),
            None => self,
        }
    }

    /// Whether the set names this contract format.
    pub const fn contains(self, format: TextureFormat) -> bool {
        match Self::bit(format) {
            Some(bit) => self.0 & bit != 0,
            None => false,
        }
    }

    /// How many lanes the set names.
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }
}

/// The two facts one provider's frame states about its three-dimensional
/// sampled arm: which lanes it carries, and how large a volume it admits.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VolumeSupport {
    /// The lanes the frame lists for the arm.
    pub lanes: VolumeLanes,
    /// The largest extent the frame admits on **each** of a volume's three
    /// axes; `0` for a frame that does not carry the section at all.
    pub window: u64,
}

impl VolumeSupport {
    /// The answer a device that states neither the lane nor the window gives,
    /// which is also what every request whose binds name no volume reads: the
    /// class gate then answers exactly as it did before this increment.
    pub const NONE: Self = Self {
        lanes: VolumeLanes::NONE,
        window: 0,
    };

    /// The contract format this device's answer states for one bind's own
    /// Vulkan view of a volume lane, or `None` when the frame does not list it.
    ///
    /// One term per lane rather than `r32_float` alone, and the terms are the
    /// contract's own [`TextureFormat`] names: the census's volumes arrive in
    /// two shapes — the single-component float lane
    /// (`texture3d<float, sample>` over `R32_SFLOAT`) and the eight-bit
    /// four-component order (`B8G8R8A8_UNORM`) — and which of them a device
    /// carries is a fact only the device states. The three-axis window beside
    /// the list is a limit rather than a second format, which is why a lane the
    /// frame lists is still refused by the same door when the bind's extents
    /// fall outside it.
    pub const fn admits(self, format: ash::vk::Format) -> Option<TextureFormat> {
        let lane = match format {
            ash::vk::Format::R32_SFLOAT => Some(TextureFormat::R32Float),
            ash::vk::Format::R8G8B8A8_UNORM => Some(TextureFormat::Rgba8Unorm),
            ash::vk::Format::B8G8R8A8_UNORM => Some(TextureFormat::Bgra8Unorm),
            ash::vk::Format::R8_UNORM => Some(TextureFormat::R8Unorm),
            ash::vk::Format::R8G8_UNORM => Some(TextureFormat::R8G8Unorm),
            ash::vk::Format::R16G16B16A16_SFLOAT => Some(TextureFormat::Rgba16Float),
            ash::vk::Format::R16_SFLOAT => Some(TextureFormat::R16Float),
            _ => None,
        };
        match lane {
            Some(lane) if self.lanes.contains(lane) => Some(lane),
            _ => None,
        }
    }

    /// Whether this window covers one volume's own three extents.
    ///
    /// One comparison per axis, exactly as the provider's own
    /// `maxImageDimension3D` rule reads: a volume is inside the window when
    /// each of its extents is, and the product is not the question — a device
    /// that admits `2048` texels per axis admits `2048 x 2048 x 2048` as far as
    /// this rule is concerned, and its own `maxResourceSize` is a different
    /// question the sampled arm never asks.
    pub const fn covers(self, extents: [u32; 3]) -> bool {
        let [width, height, depth] = extents;
        width as u64 <= self.window && height as u64 <= self.window && depth as u64 <= self.window
    }
}

/// The runtime-sampler half of the same render-sampler section (R36).
///
/// The fifth reading of the one-snapshot rule ([`stage_buffer_support`],
/// [`compute_texture_support`], [`render_texture_support`],
/// [`stage_buffer_namespace_split`]), and the one R36 needs: a pass whose
/// runtime `[[sampler(n)]]` states travel a frame is executed by the provider
/// exactly when the provider's own capability frame declares the render-sampler
/// shape at all (`supports_render_texture_sampling`, the section v70 gave the
/// tail), because the states are the pass half of a pairing whose other half is
/// that section's texture declarations — a module's texture sampled through a
/// state nobody stated is refused at admission by name
/// (`render_runtime_sampler_missing`), and a device that does not declare the
/// section is a device whose admission never states it.
///
/// The carriage itself is the codec's side and is not assumed here: E-TX4's
/// `PASS_KIND_RENDER_SAMPLERS` family (`0x15..=0x18`) writes the pass's sampler
/// block and the provider's own decoder reads it back, which the rail's tests
/// falsify by decoding a captured frame (`carried_submission`) and asserting the
/// states and their texture pairings survived the round trip. What this
/// function answers is whether the *device* declares the shape the block
/// belongs to.
///
/// `false` is the fail-closed answer, and it is what a frame written before the
/// section existed decodes to (the section is a presence tag: absent means the
/// defaults), so a caller that reads this keeps the draw on the engine under its
/// own name rather than handing the provider a pass whose states nothing
/// executes.
pub fn render_sampler_carriage(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<bool, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    Ok(decoded.supports_render_texture_sampling)
}

/// The folded-pair half of one provider capability snapshot, read back out of
/// the response frame the provider would send (`research/docs/23` §3.3,
/// E-TX9 / R33).
///
/// The fourth reading of the same one-snapshot rule ([`stage_buffer_support`],
/// [`compute_texture_support`], [`render_texture_support`]), and the one this
/// rail's folded exit needs: a `[[buffer(n)]]` argument at *both* stages of one
/// pass is a shape the canonical rail refuses when the two halves fold onto one
/// descriptor (`render_stage_buffer_layout_unsupported`), unless the provider
/// declares that it arranges the two index spaces apart — the vertex stage's
/// whole layout in the canonical namespace set
/// (`metal_api_vulkan::stage_buffer_namespace_layout`).
///
/// `true` is that declaration. `false` is the fail-closed answer, and it is
/// what a frame written before the bit existed decodes to
/// (`CAPABILITY_STAGE_BUFFER_NAMESPACE_TAIL` is a presence tag: absent means
/// the shape was never declared), so the class gate that reads this reading
/// keeps the folded pair on the engine under R31's own name rather than
/// handing the provider a shape the frame never carried.
pub fn stage_buffer_namespace_split(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<bool, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    Ok(decoded.supports_render_stage_buffer_namespace_split)
}

/// The whole-binding half of one provider capability snapshot, read back out of
/// the response frame the provider would send (`research/docs/23` §3.3,
/// E-SB3 / R48).
///
/// The same one-snapshot rule the five readings above keep, and the one this
/// rail's stage-buffer walk needs: a `[[buffer(N)]]` argument whose *reach the
/// translation could not state* — `StageBufferFootprint::Unstated`, the
/// reflection's `has_unbounded_access` / no-range-at-all — is a shape the
/// canonical contract answers in two different ways depending on the device.
/// With the arm declared the contract states
/// `FootprintProof::BindingRange` and the provider executes the bind's own
/// window whole; without it the arm is refused by name
/// (`render_stage_buffer_footprint_unsupported`) and the draw keeps the engine
/// under the census's own bucket.
///
/// `true` is the provider's declaration that it executes the arm, and it is the
/// device's own answer on the canonical side: the Vulkan rail publishes it
/// exactly when the selected device reported `robustBufferAccess` and was
/// created with it enabled, so an access past the binding is clamped (reads) or
/// discarded (writes) instead of being unruly undefined behaviour — the same
/// sentence the contract's own arm rests on.
///
/// `false` is the fail-closed reading, and it is what a frame written before
/// the bit existed decodes to (`CAPABILITY_RENDER_STAGE_BUFFER_BINDING_RANGE_TAIL`
/// is a presence tag: absent means the arm was never declared), so the class
/// gate that reads this reading keeps its own refusal sentence and slug for the
/// shape rather than handing the provider a declaration the frame never
/// carried.
pub fn stage_buffer_binding_range(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<bool, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    Ok(decoded.supports_render_stage_buffer_binding_range)
}

/// The gathered-extent half of one provider capability snapshot, read back out
/// of the response frame the provider would send (`research/docs/23` §3.3,
/// E-TX10 / R37).
///
/// The sixth reading of the same one-snapshot rule ([`stage_buffer_support`],
/// [`compute_texture_support`], [`render_texture_support`],
/// [`render_sampler_carriage`], [`stage_buffer_namespace_split`]), and the one
/// this rail's extent exit needs: a sampled texture whose bind is *another*
/// extent than the pass it is read in is a shape the two rails E ships answer
/// differently (R35) — the canonical Vulkan rail gathers a source of another
/// extent into the render area's own grid and keeps
/// `render_texture_extent_unsupported` for the owner's no-copy window alone
/// (`research/docs/23` §111, E-TX5), while `metal-api-native` refuses every
/// source of another extent under that same name.
///
/// `true` is the Vulkan rail's own declaration that it executes the host-bytes
/// half of that split — the sources a rail reads off the host, the arm R35
/// counts under `texture_extent_host_bytes`. `false` is the fail-closed answer,
/// and it is what a frame written before the bit existed decodes to (the
/// escape family E-TX9 opened carries this bit as its second in-family tag,
/// `0x00 0x02 <bool>`, so its absence reads as undeclared), so the class gate
/// that reads this reading keeps the draw on the engine under R35's own name
/// rather than handing the provider a source of another extent.
///
/// The arm the bit deliberately does *not* cover is the owner's no-copy window:
/// a borrowed window of another extent is read without any host copy at all, so
/// E's window rule answers it with code of its own and E-TX12 publishes that
/// answer as a second bit — [`render_texture_gathered_extent_no_copy`], which is
/// the reading that arm is asked for. A device that declares this reading and
/// not that one keeps the no-copy arm on the engine.
pub fn render_texture_gathered_extent(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<bool, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    Ok(decoded.supports_render_texture_gathered_extent)
}

/// The declared-superset vertex interface's half of one provider capability
/// snapshot, read back out of the response frame the provider would send
/// (`research/docs/23` §3.3, E-TX11 / R-VI1).
///
/// The seventh reading of the same one-snapshot rule ([`stage_buffer_support`],
/// [`compute_texture_support`], [`render_texture_support`],
/// [`render_sampler_carriage`], [`stage_buffer_namespace_split`],
/// [`render_texture_gathered_extent`]), and the one this rail's vertex-interface
/// exit needs: a request whose declared attribute layout names a location the
/// vertex stage never reads is a shape Metal answers — `MTLVertexDescriptor` is
/// free to name such a location, and the surplus stream is *bound and ignored*
/// — but it is also a shape one of the two rails E ships refuses by name
/// (`metal-api-vulkan`'s `validate_translated_vertex_attributes`, R-VI1's
/// `render_provider_out_of_class_vertex_interface`).
///
/// `true` is the Vulkan rail's own declaration that it executes that direction:
/// its vertex input state is built from the *declared* layout (`create_pipeline`
/// emits one `VkVertexInputAttributeDescription` per declared attribute), which
/// is exactly Metal's descriptor. `false` is the fail-closed answer, and it is
/// what a frame written before the bit existed decodes to (E-TX11 extends the
/// escape family E-TX9 opened with its third in-family tag, `0x00 0x03 <bool>`,
/// so its absence reads as undeclared), so the class gate that reads this
/// reading keeps the draw on the engine under R-VI1's own name rather than
/// handing the provider a shape the frame never carried.
///
/// The direction the bit deliberately does *not* cover is the reflected
/// superset: a location the vertex stage reads that no declared entry covers is
/// a value Metal leaves undefined, so that arm keeps its refusal on every
/// per-snapshot answer and this reading is never asked for it.
pub fn render_vertex_interface_superset(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<bool, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    Ok(decoded.supports_render_vertex_interface_superset)
}

/// The gathered extent's *no-copy* half of one provider capability snapshot,
/// read back out of the response frame the provider would send
/// (`research/docs/23` §111, E-TX12 / R40).
///
/// The sibling of [`render_texture_gathered_extent`] and the fourth tag of the
/// same escape family, and the one this rail's extent exit needs beside it: the
/// shape the two arms of R35 split is answered by *different* code on the
/// canonical side, so one bit could never state both. E-TX10's bit states the
/// arm whose bytes a rail reads off the host — a source gathered into the render
/// area's own grid. This bit states the arm with no host bytes at all: the
/// owner's no-copy window, which a registration executing the reviewed sampling
/// module's *gathered* sibling reads in place, on the device, at the
/// destination grid's own integer index (E-TX12's `OpImageFetch` with the two
/// extents as specialization constants), and which a *translated* fragment
/// stage reads at the source's own extent exactly as it reads the trace's own
/// bytes.
///
/// `true` is the Vulkan rail's own declaration that it executes that arm —
/// without a host copy of the owner's mapping, which is the whole statement of
/// the arm. `false` is the fail-closed answer, and it is what a frame written
/// before the bit existed decodes to (it is the escape family E-TX9 opened with
/// its fourth in-family tag, `0x00 0x04 <bool>`, so its absence reads as
/// undeclared), so the class gate that reads this reading keeps the draw on the
/// engine under R35's own name, sentence and route rather than handing the
/// provider a window the frame never stated it could read.
///
/// The arms the bit deliberately does *not* cover are the ones E-TX10's reading
/// states: a source a rail gathers off the host, and every source whose extent
/// *is* the pass's own. Neither is an arm this reading can widen, and a device
/// that declares only this one still keeps the host-bytes arm on the engine.
pub fn render_texture_gathered_extent_no_copy(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<bool, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    Ok(decoded.supports_render_texture_gathered_extent_no_copy)
}

/// The attachment-landing-view half of one provider capability snapshot, read
/// back out of the response frame the provider would send (`research/docs/23`
/// §115 之后的增量，E-TX13).
///
/// The ninth reading of the same one-snapshot rule ([`stage_buffer_support`],
/// [`compute_texture_support`], [`render_texture_support`],
/// [`render_sampler_carriage`], [`stage_buffer_namespace_split`],
/// [`render_texture_gathered_extent`], [`render_vertex_interface_superset`],
/// [`render_texture_gathered_extent_no_copy`]), and the one this rail's
/// landing exit needs: a record whose attachment is backed by the guest's own
/// pages, whose own load is some *other* statement of its previous contents
/// (the walk's chain value, or a `Clear`), and whose frame the guest's pages
/// are owed can state where that frame lands *without* changing where the pass
/// begins from — by carrying a **second** view declaration
/// (`StoreOp::BorrowedLanding`,
/// `metal_api_core::provider::AttachmentLandingView`) beside the attachment's
/// own.
///
/// `true` is the provider's own declaration that it executes that arm: the
/// rail resolves the second declaration through the same serial view list the
/// attachment's own comes from and copies the readback into the owner's window
/// (`metal-api-vulkan`'s `resolve_attachment_landing`). `false` is the
/// fail-closed answer, and it is what a frame written before the bit existed
/// decodes to (the second family of capability tail E-TX13 opened carries this
/// bit as `0x00 0x05 <bool>`, so its absence reads as undeclared), so the class
/// gate that reads this reading keeps the record on the engine under
/// `render_provider_out_of_class_guest_backing` rather than handing the
/// provider a store arm the frame never stated it could land.
///
/// What the bit deliberately does *not* answer is where the window is: the
/// second declaration is the seam's own cut of the surface's registered pages,
/// and a provider that declares this bit still refuses a trace whose
/// declaration is missing, is a copy arm, or does not concatenate to the
/// attachment's tightly packed extent (`landing_view_undeclared` /
/// `owned_bytes` / `staged_lease` / `render_attachment_landing_mismatch`).
pub fn render_attachment_landing_view(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<bool, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    Ok(decoded.supports_render_attachment_landing_view)
}

/// The kept-frame-landing half of one provider capability snapshot, read back
/// out of the response frame the provider would send (`research/docs/23`
/// §115 之后的增量，E-TX14/R4b).
///
/// The tenth reading of the same one-snapshot rule
/// ([`stage_buffer_support`], [`compute_texture_support`],
/// [`render_texture_support`], [`render_sampler_carriage`],
/// [`stage_buffer_namespace_split`], [`render_texture_gathered_extent`],
/// [`render_vertex_interface_superset`],
/// [`render_texture_gathered_extent_no_copy`],
/// [`render_attachment_landing_view`]), and the one this rail's delayed Store
/// tails need: a record whose frame the caller asked to keep
/// (`StoreOp::Resident`) can be delivered by a **landing-only entry** that
/// stands after the pass which kept it, so the frame reaches the owner's
/// registered window without the provider ever publishing its bytes.
///
/// `true` is the provider's own declaration that it executes that entry: it
/// resolves the kept identity out of its resident registry, the landing view
/// out of the trace's serial view list, and copies the frame into the owner's
/// pages (`metal-api-vulkan`'s `land_kept_frame_entry`). `false` is the
/// fail-closed answer and it is what a frame written before the bit existed
/// decodes to (the second family of capability tail carries this bit as
/// `0x00 0x06 <bool>`, so its absence reads as undeclared), so a class gate
/// that reads this reading keeps the tail on today's published path rather
/// than handing the provider a frame with no way back.
///
/// The bit is deliberately **not** [`render_attachment_landing_view`]'s: that
/// one answers whether a pass's own frame can land in a second declaration in
/// the same completion, while this one answers whether a *later* entry can
/// deliver a frame a completed pass kept. A rail can execute either without
/// the other, so the two are read apart.
pub fn render_kept_frame_landing(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<bool, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    Ok(decoded.supports_render_kept_frame_landing)
}

/// Whether this provider executes a render pass whose runtime sampler states
/// the **texel space** (2026-09-19, census v43's `texture_state` axis).
///
/// The eighth device answer this rail asks before the class gate, read out of
/// the capability frame rather than the in-process snapshot for the same reason
/// every answer beside it is: a bit the owner→provider frame cannot carry is a
/// bit no remote owner would ever see. The section is the tail's second
/// family's `0x00 0x08 <bool>`, so a frame written before the axis reads as
/// undeclared — and the class then keeps the census's refusal sentence and slug
/// for the shape, byte for byte, which is what makes this read a widening of
/// the class by exactly what the device states.
pub fn render_pixel_coordinate_sampler(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<bool, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    Ok(decoded.supports_render_pixel_coordinate_sampler)
}

/// Whether this provider executes a render pass whose sampled declaration is
/// the **pass-entry snapshot** of a colour attachment the same pass writes
/// (`research/docs/23` §118, E-TX15).
///
/// The eleventh device answer this rail asks before the class gate, read out of
/// the capability frame rather than the in-process snapshot for the same reason
/// every answer beside it is: a bit the owner→provider frame cannot carry is a
/// bit no remote owner would ever see. The section is the tail's second
/// family's `0x00 0x09 <bool>`, so a frame written before the arm existed reads
/// as undeclared — and the class then keeps the census's `texture_source_order`
/// refusal, slug and sentence, byte for byte.
///
/// What the bit answers is the one fact the class cannot read off the request:
/// whether the provider can take the attachment's bytes *on the device, before
/// the pass opens*, and hand them to the sampled declaration as that
/// attachment's own identity. A declaration that names the attachment's view is
/// `RenderTextureAttachmentConflict` in the contract — "anything but a race" —
/// unless the provider states that its execution order resolves the read before
/// the pass (`TextureSource::PassEntrySnapshot`,
/// `metal-api-vulkan`'s `AttachmentSnapshot`). The arm's own shape rules (the
/// identity pair of *this* pass's colour attachment, the attachment's format
/// and extent restated, a load arm that keeps prior contents, one plain
/// single-sample 2D view) are the class's, and `false` keeps the refusal for
/// every one of them.
pub fn render_pass_entry_snapshot(
    epoch: DeviceEpoch,
    capabilities: &ProviderCapabilities,
) -> Result<bool, WireDecline> {
    let decoded = capabilities_frame(epoch, capabilities)?;
    Ok(decoded.supports_render_pass_entry_snapshot)
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
