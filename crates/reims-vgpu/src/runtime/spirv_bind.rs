//! SPIR-V set-0 binding relocation for metal2vulkan + the internal Vulkan engine (Linux product).
//!
//! metal2vulkan decorates every stage independently at DescriptorSet 0, in bands
//! 32 apart. [`widen_sampled_bands`] rewrites those into the device's own, wider
//! layout once per shader — buffers `[0,32)`, textures `[32,160)`, samplers
//! `[160,192)`, ColorInput / framebuffer fetch `[192,200)` — because a 32-wide
//! texture band cannot hold the 128 indices Apple's serializer emits. The two
//! numberings and why they differ are laid out in full below the constants; read
//! that before touching a band, because reflection stays in the translator's
//! numbering while the SPIR-V moves to the device's.
//!
//! Everything after the widen is stated in device numbering. The engine builds
//! one merged set 0 and rejects duplicate bindings, so a binding two stages both
//! claim has to move. When vertex and fragment both bind the same Metal buffer
//! index, fragment buffer decorations move by [`FRAG_BUFFER_BINDING_OFFSET`]
//! (into `[256,288)`). When both stages sample textures, fragment
//! sampled-resource decorations in `[32,192)` move by
//! [`FRAG_SAMPLED_RESOURCE_BINDING_OFFSET`] (textures → `[320,448)`, samplers →
//! `[448,480)`). The ColorInput band never moves — the engine binds the input
//! attachment at its un-relocated number.
//!
//! Port of archive `reims-vgpu-backend-vulkan` `spirv.rs` relocation helpers only —
//! structural SPIR-V `OpDecorate Binding` walks, no name heuristics.

/// SPIR-V `OpDecorate` opcode.
const OP_DECORATE: u16 = 71;
const OP_TYPE_IMAGE: u16 = 25;
const OP_TYPE_SAMPLER: u16 = 26;
const OP_TYPE_SAMPLED_IMAGE: u16 = 27;
const OP_TYPE_POINTER: u16 = 32;
const OP_FUNCTION: u16 = 54;
const OP_VARIABLE: u16 = 59;
const OP_FUNCTION_CALL: u16 = 57;
const OP_IMAGE_TEXEL_POINTER: u16 = 60;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_COPY_MEMORY: u16 = 63;
const OP_COPY_MEMORY_SIZED: u16 = 64;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_IN_BOUNDS_ACCESS_CHAIN: u16 = 66;
const OP_PTR_ACCESS_CHAIN: u16 = 67;
const OP_IN_BOUNDS_PTR_ACCESS_CHAIN: u16 = 70;
const OP_COPY_OBJECT: u16 = 83;
const OP_IMAGE_READ: u16 = 98;
const OP_IMAGE_WRITE: u16 = 99;
const OP_IMAGE_QUERY_FORMAT: u16 = 101;
const OP_IMAGE_QUERY_ORDER: u16 = 102;
const OP_IMAGE_QUERY_SIZE_LOD: u16 = 103;
const OP_IMAGE_QUERY_SIZE: u16 = 104;
const OP_IMAGE_QUERY_LOD: u16 = 105;
const OP_IMAGE_QUERY_LEVELS: u16 = 106;
const OP_IMAGE_QUERY_SAMPLES: u16 = 107;
const OP_CONVERT_PTR_TO_U: u16 = 117;
const OP_PTR_CAST_TO_GENERIC: u16 = 121;
const OP_GENERIC_CAST_TO_PTR: u16 = 122;
const OP_GENERIC_CAST_TO_PTR_EXPLICIT: u16 = 123;
const OP_SELECT: u16 = 169;
const OP_ATOMIC_STORE: u16 = 228;
const OP_ATOMIC_EXCHANGE: u16 = 229;
const OP_ATOMIC_COMPARE_EXCHANGE: u16 = 230;
const OP_ATOMIC_COMPARE_EXCHANGE_WEAK: u16 = 231;
const OP_ATOMIC_I_INCREMENT: u16 = 232;
const OP_ATOMIC_I_DECREMENT: u16 = 233;
const OP_ATOMIC_I_ADD: u16 = 234;
const OP_ATOMIC_I_SUB: u16 = 235;
const OP_ATOMIC_S_MIN: u16 = 236;
const OP_ATOMIC_U_MIN: u16 = 237;
const OP_ATOMIC_S_MAX: u16 = 238;
const OP_ATOMIC_U_MAX: u16 = 239;
const OP_ATOMIC_AND: u16 = 240;
const OP_ATOMIC_OR: u16 = 241;
const OP_ATOMIC_XOR: u16 = 242;
const OP_PHI: u16 = 245;
const OP_RETURN_VALUE: u16 = 254;
const OP_ATOMIC_FLAG_TEST_AND_SET: u16 = 318;
const OP_ATOMIC_FLAG_CLEAR: u16 = 319;
const OP_CAPABILITY: u16 = 17;
// The three storage-image capability numbers, from SPIR-V's `Capability` enum.
//
// **These are the numbers the validator checks, and one of them was wrong for a
// long time without anything noticing.** `StorageImageWriteWithoutFormat` was
// spelled `34`, which is `ImageCubeArray` — so the splice that existed to make
// a format-less write legal declared an unrelated capability, the module was
// rejected exactly as if nothing had been spliced, and the rejection named the
// capability that was supposedly just added. Both x86 rails lost compute work to
// it on every boot.
//
// The test over that splice could not see it: it asserted the spliced word
// equalled `CAPABILITY_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT`, which is the
// constant compared against itself. Only the validator knows these numbers, so
// the only real check is a module that reaches it — which is what
// `required_image_capabilities` plus a driven boot now provides.
//
/// SPIR-V `Capability StorageImageWriteWithoutFormat` (writes to an `Unknown`
/// format storage image). Paired with the Vulkan feature of the same name.
const CAPABILITY_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT: u32 = 56;
/// SPIR-V `Capability StorageImageReadWithoutFormat` (reads from an `Unknown`
/// format storage image). Paired with the Vulkan feature of the same name.
const CAPABILITY_STORAGE_IMAGE_READ_WITHOUT_FORMAT: u32 = 55;
/// SPIR-V `Capability StorageImageExtendedFormats`, required by any storage
/// image whose declared format is outside the core set. Paired with the Vulkan
/// feature `shaderStorageImageExtendedFormats`.
const CAPABILITY_STORAGE_IMAGE_EXTENDED_FORMATS: u32 = 49;

// The three are distinct and none is `Shader` (1), which is the one every module
// already declares. A collision here would make a splice a silent no-op.
const _: () = assert!(
    CAPABILITY_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT != CAPABILITY_STORAGE_IMAGE_READ_WITHOUT_FORMAT
        && CAPABILITY_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT
            != CAPABILITY_STORAGE_IMAGE_EXTENDED_FORMATS
        && CAPABILITY_STORAGE_IMAGE_READ_WITHOUT_FORMAT
            != CAPABILITY_STORAGE_IMAGE_EXTENDED_FORMATS
);
/// `OpTypeImage`'s `Sampled` operand value for "used as a storage image".
///
/// Operand 7 of `OpTypeImage`; 1 means sampled, 2 means storage, 0 means either.
/// Only a storage image's format operand carries a capability requirement, so
/// this is what keeps a sampled image with an exotic format from being read as
/// one.
const IMAGE_SAMPLED_STORAGE: u32 = 2;
/// SPIR-V `Decoration Binding`.
const DECORATION_BINDING: u32 = 33;
const HEADER_WORDS: usize = 5;
/// First binding of the translator's sampled-resource band, and therefore the
/// exclusive end of its buffer band — the two are the same number because the
/// bands abut.
///
/// A second constant used to spell this `BUFFER_BINDING_LIMIT: u32 = 32` on the
/// line above, which read as Apple's buffer bind limit and was not one: that is
/// `reims_vgpu_wire::ops::bind_limit::BUFFER`, and it is **31**. Nothing here
/// bounds how many buffers a guest may bind; this only says where one band stops
/// and the next starts, which is a fact about our own translated layout.
const SAMPLED_RESOURCE_BINDING_BASE: u32 = 32;
const STORAGE_CLASS_UNIFORM_CONSTANT: u32 = 0;
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

// ---------------------------------------------------------------------------
// Two numberings, and why they are not the same one
// ---------------------------------------------------------------------------
//
// `metal2vulkan` emits its own bands, 32 apart, and they are the *input* to this
// module. They are imported from the translator rather than re-declared, so a
// change on that side fails this build instead of silently disagreeing.
//
// Those bands are too narrow: Metal's texture argument table is 128 entries and
// Apple's serializer emits up to that (`bind_limit::TEXTURE`), so a texture at
// index 40 would decorate binding 72 — the same number the translator gives
// sampler 8. The device therefore uses a *wider* layout, and
// [`widen_sampled_bands`] rewrites the translator's output into it once per
// shader. Textures do not move (their base is the same in both), so every
// consumer keyed on `TEXTURE_BINDING_BASE + metal_index` is unaffected; the
// sampler and ColorInput bands move up out of the texture band's way.
//
//   class        translator emits   device uses      width
//   buffers      [0, 32)            [0, 32)          32   (Metal's table is 31)
//   textures     [32, 64)           [32, 160)        128  (Metal's table, exactly)
//   samplers     [64, 96)           [160, 192)       32   (Metal's table is 16)
//   ColorInput   [96, 104)          [192, 200)       8    (MRT ≤ 8)
//
// The rewrite is keyed on the SPIR-V *type* behind each variable
// ([`variable_classes`]), never on the number, which is what lets it separate a
// texture at 72 from a sampler at 72. That also means it repairs a module in
// which the translator gave both the same binding: two variables that collided
// as one number come out as two.
pub use metal2vulkan::reflect::{
    COLOR_INPUT_BINDING_BASE as M2V_COLOR_INPUT_BINDING_BASE,
    SAMPLER_BINDING_BASE as M2V_SAMPLER_BINDING_BASE,
    TEXTURE_BINDING_BASE as M2V_TEXTURE_BINDING_BASE,
};

/// Device texture band base (Metal texture index N → binding 32+N).
///
/// Equal to the translator's own base by construction, so no texture decoration
/// is ever rewritten and every reflection lookup keyed on this number stays
/// valid without translation.
pub const TEXTURE_BINDING_BASE: u32 = M2V_TEXTURE_BINDING_BASE;
/// Device sampler band base (Metal sampler index N → binding 160+N).
///
/// 160 rather than the translator's 64, so the texture band below it is 128 wide
/// — exactly Metal's texture argument table, and exactly what Apple's serializer
/// is entitled to emit.
pub const SAMPLER_BINDING_BASE: u32 = 160;
/// Device ColorInput band base: `air.render_target` INPUT params (framebuffer
/// fetch, `dest_N`) emit `SubpassData` images, which the translator numbers from
/// [`M2V_COLOR_INPUT_BINDING_BASE`]. The band (MRT ≤ 8) must survive BOTH
/// fragment relocations unchanged — the engine binds the input attachment by
/// this number. m2v-synthesized constexpr samplers currently also land here;
/// they are unbindable either way, so preserving the band never makes them worse.
pub const COLOR_INPUT_BINDING_BASE: u32 = 192;

/// How far [`widen_sampled_bands`] moves a sampler or ColorInput decoration.
///
/// One offset for both bands, so their spacing is preserved and the widen is a
/// single translation rather than a per-class table.
pub const SAMPLED_TAIL_WIDEN_OFFSET: u32 = SAMPLER_BINDING_BASE - M2V_SAMPLER_BINDING_BASE;
const _: () = assert!(
    COLOR_INPUT_BINDING_BASE - M2V_COLOR_INPUT_BINDING_BASE == SAMPLED_TAIL_WIDEN_OFFSET,
    "one offset moves both tail bands, so their spacing must be preserved"
);
// The texture band is exactly Metal's argument table, which is the whole point
// of the widening: `runtime::draw::MAX_TEXTURE_BIND_SLOTS` reads its value from
// this subtraction, and `runtime::exec` pins that against Apple's own table.
const _: () = assert!(TEXTURE_BINDING_BASE == M2V_TEXTURE_BINDING_BASE);
const _: () = assert!(SAMPLER_BINDING_BASE - TEXTURE_BINDING_BASE == 128);

/// Fragment buffer band destination offset (`[0,32)` → `[256,288)`) — starts
/// past every un-relocated band (which now end at [`COLOR_INPUT_BINDING_BASE`]
/// + 8 = 200) and ends before the relocated sampled bands.
pub const FRAG_BUFFER_BINDING_OFFSET: u32 = 256;
/// Fragment sampled-resource destination offset (textures/samplers
/// `[32,192)` → `+288`, so textures land in `[320,448)` and samplers in
/// `[448,480)`), clear of the relocated fragment buffer band.
pub const FRAG_SAMPLED_RESOURCE_BINDING_OFFSET: u32 = 288;
/// Exclusive upper bound of the sampled-resource source band relocated by
/// [`offset_fragment_sampled_resource_bindings`]: textures `[32,160)` + samplers
/// `[160,192)`. Bindings at [`COLOR_INPUT_BINDING_BASE`] and above stay in place.
const SAMPLED_RESOURCE_BINDING_LIMIT: u32 = COLOR_INPUT_BINDING_BASE;

// The band map, pinned at build time rather than in a test, because a band that
// no longer holds what a guest can send is a silently mis-bound descriptor and
// not a slow path. Each of these is one relation between two constants that can
// move independently.
//
// Every Metal texture index Apple's serializer can emit lands inside the texture
// band, and the last one is its last slot — no waste, no overflow.
const _: () = assert!(
    TEXTURE_BINDING_BASE + reims_vgpu_wire::ops::bind_limit::TEXTURE == SAMPLER_BINDING_BASE
);
// Every Metal sampler index it can emit lands inside the sampler band.
const _: () = assert!(
    SAMPLER_BINDING_BASE + reims_vgpu_wire::ops::bind_limit::SAMPLER <= COLOR_INPUT_BINDING_BASE
);
// The relocated fragment buffer band starts past every un-relocated band,
// including the 8-entry ColorInput band at the top of them.
const _: () = assert!(FRAG_BUFFER_BINDING_OFFSET >= COLOR_INPUT_BINDING_BASE + 8);
// ...and ends before the relocated fragment sampled bands begin.
const _: () = assert!(
    SAMPLED_RESOURCE_BINDING_BASE - 1 + FRAG_BUFFER_BINDING_OFFSET
        < TEXTURE_BINDING_BASE + FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
);
// The widest relocated binding still fits a `u32`.
const _: () = assert!(
    (SAMPLED_RESOURCE_BINDING_LIMIT - 1).checked_add(FRAG_SAMPLED_RESOURCE_BINDING_OFFSET)
        .is_some()
);

/// Image dimensionality declared by a translated SPIR-V sampled-image binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampledImageKind {
    D1,
    D1Array,
    D2,
    D2Array,
    D3,
    Cube,
    CubeArray,
}

/// Sampled-vs-storage class of a texture binding, derived from the translator's
/// reflection (`TextureShape.writable`): a writable texture is a storage image, a
/// read/sample texture is a sampled image. The declared Metal access qualifier is
/// authoritative, so this is exact at translate time — there is no `Unknown`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageAccess {
    Sampled,
    Storage,
}

/// Content access proven from the SPIR-V use graph for one storage image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StorageImageAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
    /// The image object escaped or participated in an operation whose content
    /// access cannot be classified safely.
    Unknown,
    /// More than one image variable declares the same binding.
    AmbiguousBinding,
}

/// Explicit storage-image texel format declared by `OpTypeImage`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageFormat {
    Rgba32Float,
    Rgba16Float,
    R16Float,
    Rgba16Uint,
    Rgba8Uint,
    Rgba8Sint,
    Rgba8Unorm,
    Rg16Float,
    R8Unorm,
    Rg8Unorm,
    Rgba32Uint,
    /// Single-channel 32-bit float (SPIR-V `R32f`, enum value 3). This is what
    /// `metal2vulkan` declares for a generic `texture2d<float, access::write>`
    /// (`storage_format_from_name`: a `<float` scalar lowers to `R32f`), so it
    /// arrives on the wire constantly — a full-width float write target whose
    /// real texel format is whatever guest surface gets bound. Like [`Self::R32ui`]
    /// it is specialized against that surface before use; leaving it undecoded
    /// made every such dispatch `Unsupported(3)` and dropped it.
    R32Float,
    /// Single-channel 32-bit uint (SPIR-V `R32ui`). Not emitted by the
    /// translator (which declares `Rgba8ui` for a generic `texture2d<uint,
    /// write>`); the device *specializes* a storage image to this format when
    /// the bound guest surface is `MTLPixelFormatR32Uint`, so the view is
    /// `VK_FORMAT_R32_UINT` and a written `uint4`'s `.x` lane is the full u32
    /// (a `Rgba8ui` raw view would keep only the low byte of each lane).
    R32ui,
    /// SPIR-V `Unknown` storage format (enum value 0): the image carries no
    /// declared texel format, so its `VkImageView` may be any compatible format
    /// and the GPU converts written vec4s to that view's channel order. Reads
    /// need `StorageImageReadWithoutFormat`, writes `StorageImageWriteWithoutFormat`.
    /// The device targets this deliberately for a guest `BGRA8Unorm` storage
    /// surface (viewed `B8G8R8A8_UNORM`), which SPIR-V cannot name directly.
    Unknown,
    /// An explicit format outside the product engine's supported surface.
    Unsupported(u32),
}

impl ImageFormat {
    fn from_raw(raw: u32) -> Self {
        match raw {
            0 => Self::Unknown,
            1 => Self::Rgba32Float,
            2 => Self::Rgba16Float,
            3 => Self::R32Float,
            4 => Self::Rgba8Unorm,
            7 => Self::Rg16Float,
            9 => Self::R16Float,
            13 => Self::Rg8Unorm,
            15 => Self::R8Unorm,
            23 => Self::Rgba8Sint,
            30 => Self::Rgba32Uint,
            31 => Self::Rgba16Uint,
            32 => Self::Rgba8Uint,
            33 => Self::R32ui,
            _ => Self::Unsupported(raw),
        }
    }

    fn raw(self) -> u32 {
        match self {
            Self::Rgba32Float => 1,
            Self::Rgba16Float => 2,
            Self::R32Float => 3,
            Self::Rgba8Unorm => 4,
            Self::Rg16Float => 7,
            Self::R16Float => 9,
            Self::Rg8Unorm => 13,
            Self::R8Unorm => 15,
            Self::Rgba8Sint => 23,
            Self::Rgba32Uint => 30,
            Self::Rgba16Uint => 31,
            Self::Rgba8Uint => 32,
            Self::R32ui => 33,
            Self::Unknown => 0,
            Self::Unsupported(raw) => raw,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageFormatSpecializeError {
    MalformedModule,
    MissingBinding(u32),
    AmbiguousBinding(u32),
    /// An `OpLoad` still declares a type its variable no longer points at, so
    /// the module this device assembled is not valid SPIR-V.
    ///
    /// Only reachable through the clone path — see `specialize_image_formats`
    /// — and only if something there consumed a repointed variable in a way
    /// the retype pass does not know how to follow. It is a refusal rather than
    /// a repair because the alternative is handing the driver a module it may
    /// not survive: an NVIDIA SPIR-V compiler segmentation-faults inside
    /// `vkCreateComputePipelines` on exactly this defect, which ends the VM
    /// process. A guest authors its own kernels, so this must be a decline.
    LoadTypeMismatch { pointer: u32, declared: u32 },
}

impl crate::observe::Decline for ImageFormatSpecializeError {
    /// The slug table used to live at the *caller*, as a `match` inside
    /// `compute_exec.rs` mapping each variant to a string. That works until a
    /// second caller appears and writes its own table — which is how one check
    /// ends up with two names. It belongs to the type.
    fn slug(&self) -> &'static str {
        match self {
            Self::MalformedModule => "spirv_format_specialize_malformed",
            Self::MissingBinding(_) => "spirv_format_specialize_missing_binding",
            Self::AmbiguousBinding(_) => "spirv_format_specialize_ambiguous_binding",
            Self::LoadTypeMismatch { .. } => "spirv_format_specialize_load_type_mismatch",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::MalformedModule => Vec::new(),
            Self::MissingBinding(b) | Self::AmbiguousBinding(b) => {
                vec![("binding", b.to_string())]
            }
            Self::LoadTypeMismatch { pointer, declared } => vec![
                ("pointer", pointer.to_string()),
                ("declared", declared.to_string()),
            ],
        }
    }
}

/// Write access proven from the SPIR-V pointer-use graph for one storage buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferAccess {
    ReadOnly,
    Writable,
    /// A pointer escapes through a function call/return, so local provenance
    /// cannot prove whether the callee writes it.
    PointerEscape,
    /// More than one storage-buffer variable declares the same binding.
    AmbiguousBinding,
}

/// One instruction's header, decoded once.
///
/// `at` indexes the header word itself, so an operand `n` is `words[at + n]` and
/// the instruction's last word is `words[at + word_count - 1]`.
#[derive(Clone, Copy)]
struct Instruction {
    opcode: u16,
    word_count: usize,
    at: usize,
}

/// Decode every instruction header in a module body, or `None` if the stream is
/// not one.
///
/// # Why the scans below take the output of this rather than `&[u32]`
///
/// A `&[Instruction]` only exists if the whole stream walked cleanly, so a scan
/// holding one may index `words[at + n]` for any `n < word_count` without a
/// bounds check, and cannot spin. That is not a new rule — it is the rule the
/// provenance scans have always run under, taken from `descriptor_root` having
/// been called first and having returned `Some`. Nothing recorded the
/// dependency: `propagate_derived` and both escape scans re-walked the raw
/// `&[u32]` with no guard of their own, and were correct only because that one
/// guard, in another function, had already rejected every stream that would
/// break them.
///
/// The cost of leaving it implicit is not a style point. `words[at + 1 ..
/// at + word_count]` on a truncated final instruction panics, and a zero word
/// count makes `i += word_count` spin forever — a device abort and a wedged
/// guest, from a reflector whose contract is to fail closed. Making the
/// entitlement a type means a scan cannot be reached without it.
///
/// Decoding once also removes the re-decode from `propagate_derived`, which
/// re-walks to a fixpoint and used to re-split every header on every pass.
///
/// # What a driven boot says about it, and what it cannot
///
/// Driven x86/PCI boot on the consolidated walk (web-content probe, 10 captures):
/// `linux_m2v_async` 160, so 160 real guest shaders were translated and reflected
/// through it; `m2v_reflect_malformed` and `spirv_reloc_unclassified_binding` both
/// absent; 10 of 10 regions measured their declared colour. Against the boot
/// before it (158 translations) the fail-channel reason ranking gained no new
/// entry. That is the regression evidence that matters here — the reflectors'
/// answers for well-formed modules are what must not move, and 160 shaders is a
/// wider sample than the unit tests reach.
///
/// It says **nothing about the malformed cases**, and no boot can. This SPIR-V is
/// the translator's own output rather than anything a guest sends, so a guest
/// cannot drive a stream that fails this parse; only a translator bug produces
/// one, which is exactly why the failure had to stop being a panic. The coverage
/// for those is `tests::a_module_that_does_not_walk_reflects_nothing`.
fn instructions(words: &[u32]) -> Option<Vec<Instruction>> {
    if words.len() < HEADER_WORDS {
        return None;
    }
    let mut out = Vec::new();
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        if word_count == 0 || i + word_count > words.len() {
            return None;
        }
        out.push(Instruction {
            opcode: (words[i] & 0xffff) as u16,
            word_count,
            at: i,
        });
        i += word_count;
    }
    Some(out)
}

/// The one descriptor variable declaring `wanted_binding` in `storage_class`,
/// with the module's id bound.
///
/// `OpDecorate Binding` and `OpVariable` are the only two declarations either
/// provenance question in this module needs, and both ask for exactly one
/// match: two variables sharing a binding means neither can be reflected, which
/// is `Root::Ambiguous`.
///
/// `None` is a module this reflector cannot parse at all — a zero id bound, or
/// no variable on that binding. A stream whose instructions do not walk cleanly
/// never reaches here at all: [`instructions`] rejects it, and its `None` is
/// this one. Every case must fail closed rather than reflect a guess.
enum Root {
    One { id: usize, bound: usize },
    Ambiguous,
}

fn descriptor_root(
    words: &[u32],
    instrs: &[Instruction],
    wanted_binding: u32,
    storage_class: u32,
) -> Option<Root> {
    let bound = *words.get(3)? as usize;
    if bound == 0 {
        return None;
    }
    let mut bindings = vec![None; bound];
    let mut storage = vec![None; bound];
    for &Instruction {
        opcode,
        word_count,
        at: i,
    } in instrs
    {
        match opcode {
            OP_DECORATE if word_count >= 4 && words[i + 2] == DECORATION_BINDING => {
                let id = words[i + 1] as usize;
                if id < bound {
                    bindings[id] = Some(words[i + 3]);
                }
            }
            OP_VARIABLE if word_count >= 4 => {
                let id = words[i + 2] as usize;
                if id < bound {
                    storage[id] = Some(words[i + 3]);
                }
            }
            _ => {}
        }
    }
    let mut roots = bindings.iter().enumerate().filter_map(|(id, binding)| {
        (*binding == Some(wanted_binding) && storage[id] == Some(storage_class)).then_some(id)
    });
    let id = roots.next()?;
    Some(if roots.next().is_some() {
        Root::Ambiguous
    } else {
        Root::One { id, bound }
    })
}

/// Mark every id whose value derives from an already-marked id, to a fixpoint.
///
/// Both provenance questions here have the same shape — seed a set of ids, then
/// re-walk the instruction stream marking any result built from a marked
/// operand, until a pass changes nothing — and differ only in which opcodes
/// propagate. `propagates` is that difference: given an instruction, it answers
/// whether that instruction's result derives from something already marked.
///
/// The opcodes handled here are the ones that merge or rename an SSA value
/// without regard to what kind of value it is: `OpCopyObject` renames,
/// `OpSelect` and `OpPhi` merge. Provenance flows through all three for a
/// pointer and for an image alike, so they are not the caller's business.
///
/// The `word_count >= 3` guard is what makes a result id exist to mark; an
/// instruction shorter than that has no result operand.
fn propagate_derived(
    words: &[u32],
    instrs: &[Instruction],
    bound: usize,
    seed: Option<usize>,
    propagates: impl Fn(u16, usize, usize, &[bool]) -> bool,
) -> Vec<bool> {
    let mut derived = vec![false; bound];
    if let Some(root) = seed {
        derived[root] = true;
    }
    loop {
        let mut changed = false;
        for &Instruction {
            opcode,
            word_count,
            at: i,
        } in instrs
        {
            let marked = |id: u32| derived.get(id as usize).copied() == Some(true);
            let result_from = match opcode {
                OP_COPY_OBJECT if word_count >= 4 => marked(words[i + 3]),
                OP_SELECT if word_count >= 6 => marked(words[i + 4]) || marked(words[i + 5]),
                OP_PHI if word_count >= 5 => (i + 3..i + word_count)
                    .step_by(2)
                    .any(|at| marked(words[at])),
                _ => propagates(opcode, word_count, i, &derived),
            };
            if result_from && word_count >= 3 {
                let result = words[i + 2] as usize;
                if result < bound && !derived[result] {
                    derived[result] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    derived
}

/// Reflect whether a storage-buffer descriptor can be written by the module.
///
/// Pointer provenance follows the SPIR-V operations that can preserve a buffer
/// pointer (`AccessChain`, `CopyObject`, `Select`, and `Phi`). Stores, copy
/// destinations, and atomics make the binding writable. Pointer calls/returns
/// fail closed as unknown; this deliberately avoids inferring mutability from
/// debug names, guest object ids, or corpus-specific function names.
///
/// The root pointer is seeded directly, which is safe here only because the
/// escape scan below enumerates the opcodes it cares about. `storage_image_access`
/// cannot seed its root for exactly that reason — see the note there.
pub fn buffer_access(words: &[u32], wanted_binding: u32) -> Option<BufferAccess> {
    let instrs = instructions(words)?;
    let (root, bound) = match descriptor_root(
        words,
        &instrs,
        wanted_binding,
        STORAGE_CLASS_STORAGE_BUFFER,
    )? {
        Root::One { id, bound } => (id, bound),
        Root::Ambiguous => return Some(BufferAccess::AmbiguousBinding),
    };

    let derived = propagate_derived(
        words,
        &instrs,
        bound,
        Some(root),
        |opcode, word_count, i, derived| {
            let marked = |id: u32| derived.get(id as usize).copied() == Some(true);
            match opcode {
                // Both families take the base pointer at operand 3 and yield another
                // pointer to the same buffer.
                OP_ACCESS_CHAIN
                | OP_IN_BOUNDS_ACCESS_CHAIN
                | OP_PTR_ACCESS_CHAIN
                | OP_IN_BOUNDS_PTR_ACCESS_CHAIN
                | OP_PTR_CAST_TO_GENERIC
                | OP_GENERIC_CAST_TO_PTR
                | OP_GENERIC_CAST_TO_PTR_EXPLICIT
                    if word_count >= 4 =>
                {
                    marked(words[i + 3])
                }
                _ => false,
            }
        },
    );

    let is_derived = |id: u32| derived.get(id as usize).copied() == Some(true);
    let mut unknown = false;
    for &Instruction {
        opcode,
        word_count,
        at: i,
    } in &instrs
    {
        let writable = match opcode {
            OP_STORE | OP_COPY_MEMORY | OP_COPY_MEMORY_SIZED | OP_ATOMIC_STORE
                if word_count >= 2 =>
            {
                is_derived(words[i + 1])
            }
            OP_ATOMIC_EXCHANGE
            | OP_ATOMIC_COMPARE_EXCHANGE
            | OP_ATOMIC_COMPARE_EXCHANGE_WEAK
            | OP_ATOMIC_I_INCREMENT
            | OP_ATOMIC_I_DECREMENT
            | OP_ATOMIC_I_ADD
            | OP_ATOMIC_I_SUB
            | OP_ATOMIC_S_MIN
            | OP_ATOMIC_U_MIN
            | OP_ATOMIC_S_MAX
            | OP_ATOMIC_U_MAX
            | OP_ATOMIC_AND
            | OP_ATOMIC_OR
            | OP_ATOMIC_XOR
            | OP_ATOMIC_FLAG_TEST_AND_SET
                if word_count >= 4 =>
            {
                is_derived(words[i + 3])
            }
            OP_ATOMIC_FLAG_CLEAR if word_count >= 2 => is_derived(words[i + 1]),
            _ => false,
        };
        if writable {
            return Some(BufferAccess::Writable);
        }
        if opcode == OP_FUNCTION_CALL
            && word_count >= 5
            && words[i + 4..i + word_count].iter().copied().any(is_derived)
        {
            unknown = true;
        }
        if opcode == OP_RETURN_VALUE && word_count >= 2 && is_derived(words[i + 1]) {
            unknown = true;
        }
        if opcode == OP_CONVERT_PTR_TO_U && word_count >= 4 && is_derived(words[i + 3]) {
            unknown = true;
        }
    }
    Some(if unknown {
        BufferAccess::PointerEscape
    } else {
        BufferAccess::ReadOnly
    })
}

/// Reflect whether a storage image consumes its pre-dispatch contents.
///
/// This follows `OpLoad` from the descriptor variable through the SSA image
/// operations that preserve identity. `OpImageRead` and `OpImageWrite` then
/// provide the content-access contract. Queries do not consume texels;
/// pointer/image escapes fail closed as [`StorageImageAccess::Unknown`].
/// The tracked set is image *values*, so the root variable is deliberately not
/// seeded: `OpLoad` from it is what produces the first image value. Seeding the
/// variable id would also be unsound here, because the escape scan below ends in
/// a catch-all — `OpDecorate root Binding N` and `OpEntryPoint`'s interface list
/// both name the variable, and either would then read as an escape and force
/// every storage image to `Unknown`. `buffer_access` seeds its root only because
/// its scan enumerates instead.
pub fn storage_image_access(words: &[u32], wanted_binding: u32) -> Option<StorageImageAccess> {
    let instrs = instructions(words)?;
    let (root, bound) = match descriptor_root(
        words,
        &instrs,
        wanted_binding,
        STORAGE_CLASS_UNIFORM_CONSTANT,
    )? {
        Root::One { id, bound } => (id, bound),
        Root::Ambiguous => return Some(StorageImageAccess::AmbiguousBinding),
    };

    let derived = propagate_derived(
        words,
        &instrs,
        bound,
        None,
        |opcode, word_count, i, _derived| {
            opcode == OP_LOAD && word_count >= 4 && words[i + 3] as usize == root
        },
    );

    let is_derived = |id: u32| derived.get(id as usize).copied() == Some(true);
    let mut read = false;
    let mut write = false;
    let mut unknown = false;
    for &Instruction {
        opcode,
        word_count,
        at: i,
    } in &instrs
    {
        match opcode {
            OP_IMAGE_READ if word_count >= 5 && is_derived(words[i + 3]) => read = true,
            OP_IMAGE_WRITE if word_count >= 4 && is_derived(words[i + 1]) => write = true,
            OP_IMAGE_QUERY_FORMAT
            | OP_IMAGE_QUERY_ORDER
            | OP_IMAGE_QUERY_SIZE_LOD
            | OP_IMAGE_QUERY_SIZE
            | OP_IMAGE_QUERY_LOD
            | OP_IMAGE_QUERY_LEVELS
            | OP_IMAGE_QUERY_SAMPLES => {
                // Shape/format queries do not consume image contents.
            }
            OP_IMAGE_TEXEL_POINTER if word_count >= 5 && is_derived(words[i + 3]) => unknown = true,
            OP_FUNCTION_CALL
                if word_count >= 5
                    && words[i + 4..i + word_count].iter().copied().any(is_derived) =>
            {
                unknown = true
            }
            OP_RETURN_VALUE if word_count >= 2 && is_derived(words[i + 1]) => unknown = true,
            OP_LOAD | OP_COPY_OBJECT | OP_SELECT | OP_PHI => {}
            _ if words[i + 1..i + word_count].iter().copied().any(is_derived) => unknown = true,
            _ => {}
        }
    }
    Some(if unknown || (!read && !write) {
        StorageImageAccess::Unknown
    } else if read && write {
        StorageImageAccess::ReadWrite
    } else if read {
        StorageImageAccess::ReadOnly
    } else {
        StorageImageAccess::WriteOnly
    })
}

/// What a validator said about a module this device was about to hand a driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpirvValidation {
    /// The module is valid as far as the validator can tell — including the
    /// case where no validator is installed, which is not evidence of anything
    /// and must not become a refusal.
    Accepted,
    /// The validator rejected it. Carries its first line, which names the
    /// instruction.
    Rejected(String),
}

/// Ask a validator whether `words` is a module a driver can be given.
///
/// # Why this is not the driver's job
///
/// It is, and that is the problem. A Vulkan driver is entitled to undefined
/// behaviour on invalid SPIR-V, and NVIDIA's takes it: a module `metal2vulkan`
/// produced for a macOS 14 guest's compositor kernel segmentation-faults inside
/// its SPIR-V compiler and ends the QEMU process — the guest, the device and
/// every other rail with it. A guest authors its own kernels, so "the module
/// was invalid" has to be a declined dispatch and can never be a dead VM.
///
/// `metal2vulkan`'s library entry point says so itself: it returns assembled
/// words and states that the caller validates. This device is that caller, and
/// it validates *after* its own edits — the format specialization and the
/// capability injection both rewrite the module after the translator has
/// finished with it, so validating the translator's output would leave this
/// device's own contribution unchecked.
///
/// # When there is no validator
///
/// The check is external (`spirv-val`, from SPIRV-Tools, which
/// `vm/boot-x86.sh` already requires and which `metal2vulkan` spawns during
/// translation). A host without it gets [`SpirvValidation::Accepted`] and one
/// fail-log line: an absent instrument is not a verdict, and refusing every
/// dispatch because a developer tool is missing would be the widening this
/// device is not allowed to do in either direction.
pub fn validate(words: &[u32]) -> SpirvValidation {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    // A private directory per call, because `spirv_val_bytes` writes a **fixed**
    // file name inside whatever directory it is given. Handed the shared
    // `/tmp`, two concurrent validations — this device's, or one of
    // `metal2vulkan`'s own async translations — write and delete the same path,
    // and the loser validates bytes it did not produce or finds no file at all.
    // Measured: three modules on a working macos-13 boot rejected that way,
    // which would have cost the guest three shaders to fix a crash on a
    // different rail.
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "reims-vgpu-spirv-val-{}-{seq}",
        std::process::id()
    ));
    if let Err(e) = std::fs::create_dir_all(&dir) {
        if crate::observe::first_sight("spirv_val_no_tmp", 0) {
            crate::observe::fail(format!("spirv_validate reason=validator_unavailable detail={e}"));
        }
        return SpirvValidation::Accepted;
    }
    let verdict = metal2vulkan::tools::spirv_val_bytes(&bytes, &dir);
    let _ = std::fs::remove_dir_all(&dir);
    match verdict {
        Ok(()) => SpirvValidation::Accepted,
        Err(why) => {
            // A missing or unrunnable tool reads as an error from the same call
            // as a rejected module, and the two must not be confused: one is a
            // module that would take the process down, the other is a laptop
            // without SPIRV-Tools installed.
            if why.contains("No such file") || why.contains("spawn") || why.contains("not found") {
                if crate::observe::first_sight("spirv_val_absent", 0) {
                    crate::observe::fail(format!(
                        "spirv_validate reason=validator_unavailable detail={}",
                        why.lines().next().unwrap_or("")
                    ));
                }
                return SpirvValidation::Accepted;
            }
            // The whole message on one line. Its first line is the wrapper's
            // own "spirv-val failed:" and the validator's diagnosis — the part
            // that names the instruction — is on the ones after it, so keeping
            // only the first is how a rejection reads as having no reason.
            let flattened: Vec<&str> = why.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
            SpirvValidation::Rejected(flattened.join(" | "))
        }
    }
}

/// Reflect the explicit texel format for one image descriptor binding.
///
/// The SPIR-V image-format operand is structural shader ABI. It is independent
/// of debug names and may intentionally differ from the guest texture's Metal
/// pixel format when the shader uses a raw integer view.
pub fn image_format(words: &[u32], wanted_binding: u32) -> Option<ImageFormat> {
    let bound = *words.get(3)? as usize;
    if words.len() < HEADER_WORDS || bound == 0 {
        return None;
    }
    let mut bindings = vec![None; bound];
    let mut pointer_pointee = vec![None; bound];
    let mut variable_type = vec![None; bound];
    let mut formats = vec![None; bound];

    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word0 = words[i];
        let word_count = (word0 >> 16) as usize;
        let opcode = (word0 & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            return None;
        }
        match opcode {
            OP_DECORATE if word_count >= 4 && words[i + 2] == DECORATION_BINDING => {
                let id = words[i + 1] as usize;
                if id < bound {
                    bindings[id] = Some(words[i + 3]);
                }
            }
            OP_TYPE_IMAGE if word_count >= 9 => {
                let id = words[i + 1] as usize;
                let raw = words[i + 8];
                let format = ImageFormat::from_raw(raw);
                if id < bound {
                    formats[id] = Some(format);
                }
            }
            OP_TYPE_POINTER if word_count >= 4 => {
                let id = words[i + 1] as usize;
                if id < bound {
                    pointer_pointee[id] = Some(words[i + 3] as usize);
                }
            }
            OP_VARIABLE if word_count >= 4 => {
                let id = words[i + 2] as usize;
                if id < bound {
                    variable_type[id] = Some(words[i + 1] as usize);
                }
            }
            _ => {}
        }
        i += word_count;
    }

    bindings.iter().enumerate().find_map(|(variable, binding)| {
        if *binding != Some(wanted_binding) {
            return None;
        }
        let pointer = variable_type[variable]?;
        let image = pointer_pointee.get(pointer).copied().flatten()?;
        formats.get(image).copied().flatten()
    })
}

/// Specialize storage-image formats from runtime resource ABI, by binding.
///
/// Metal carries the concrete pixel format on the bound texture rather than in
/// the AIR function type. SPIR-V requires that format in `OpTypeImage`, so the
/// product Vulkan path patches only that structural operand after resolving the
/// guest texture. When multiple bindings share one translated image type but
/// resolve to different runtime formats, the helper clones only the image and
/// UniformConstant pointer types and retargets the affected variables.
pub fn specialize_image_formats(
    words: &mut Vec<u32>,
    requested: &[(u32, ImageFormat)],
) -> Result<usize, ImageFormatSpecializeError> {
    let bound = *words
        .get(3)
        .ok_or(ImageFormatSpecializeError::MalformedModule)? as usize;
    if words.len() < HEADER_WORDS || bound == 0 {
        return Err(ImageFormatSpecializeError::MalformedModule);
    }
    let mut bindings = vec![None; bound];
    let mut pointer_pointee = vec![None; bound];
    let mut variable_type = vec![None; bound];
    let mut image_format_word = vec![None; bound];
    let mut image_instruction = vec![None; bound];
    let mut pointer_instruction = vec![None; bound];
    let mut variable_type_word = vec![None; bound];
    let mut insert_at = None;
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = (words[i] & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            return Err(ImageFormatSpecializeError::MalformedModule);
        }
        match opcode {
            OP_DECORATE if word_count >= 4 && words[i + 2] == DECORATION_BINDING => {
                let id = words[i + 1] as usize;
                if id < bound {
                    bindings[id] = Some(words[i + 3]);
                }
            }
            OP_TYPE_IMAGE if word_count >= 9 => {
                let id = words[i + 1] as usize;
                if id < bound {
                    image_format_word[id] = Some(i + 8);
                    image_instruction[id] = Some((i, word_count));
                }
            }
            OP_TYPE_POINTER if word_count >= 4 => {
                let id = words[i + 1] as usize;
                if id < bound {
                    pointer_pointee[id] = Some(words[i + 3] as usize);
                    pointer_instruction[id] = Some((i, word_count));
                }
            }
            OP_VARIABLE if word_count >= 4 => {
                insert_at.get_or_insert(i);
                let id = words[i + 2] as usize;
                if id < bound {
                    variable_type[id] = Some(words[i + 1] as usize);
                    variable_type_word[id] = Some(i + 1);
                }
            }
            OP_FUNCTION => {
                insert_at.get_or_insert(i);
            }
            _ => {}
        }
        i += word_count;
    }

    let mut requested_by_variable = std::collections::BTreeMap::<usize, ImageFormat>::new();
    let mut touched_image_types = std::collections::BTreeSet::<usize>::new();
    for &(wanted_binding, format) in requested {
        let mut variables = bindings
            .iter()
            .enumerate()
            .filter_map(|(id, binding)| (*binding == Some(wanted_binding)).then_some(id));
        let variable = variables
            .next()
            .ok_or(ImageFormatSpecializeError::MissingBinding(wanted_binding))?;
        if variables.next().is_some() {
            return Err(ImageFormatSpecializeError::AmbiguousBinding(wanted_binding));
        }
        let pointer = variable_type[variable]
            .ok_or(ImageFormatSpecializeError::MissingBinding(wanted_binding))?;
        let image_type = pointer_pointee
            .get(pointer)
            .copied()
            .flatten()
            .ok_or(ImageFormatSpecializeError::MissingBinding(wanted_binding))?;
        if image_format_word
            .get(image_type)
            .copied()
            .flatten()
            .is_none()
        {
            return Err(ImageFormatSpecializeError::MissingBinding(wanted_binding));
        }
        requested_by_variable.insert(variable, format);
        touched_image_types.insert(image_type);
    }

    let mut changed = 0;
    let mut next_id = bound as u32;
    let mut extra = Vec::new();
    // Variables whose pointer type was cloned, and the image type they now
    // name. Their loads are repaired once at the end, after the splice.
    let mut retyped = std::collections::BTreeMap::<u32, u32>::new();
    for image_type in touched_image_types {
        let at = image_format_word[image_type].expect("validated image type");
        let original = ImageFormat::from_raw(words[at]);
        let mut variables = Vec::new();
        for (variable, pointer) in variable_type.iter().enumerate() {
            let Some(pointer) = *pointer else {
                continue;
            };
            if pointer_pointee.get(pointer).copied().flatten() == Some(image_type) {
                variables.push((
                    variable,
                    pointer,
                    requested_by_variable
                        .get(&variable)
                        .copied()
                        .unwrap_or(original),
                ));
            }
        }
        let keep = variables
            .iter()
            .find_map(|(variable, _, format)| {
                (!requested_by_variable.contains_key(variable)).then_some(*format)
            })
            .or_else(|| variables.first().map(|(_, _, format)| *format))
            .ok_or(ImageFormatSpecializeError::MalformedModule)?;
        if words[at] != keep.raw() {
            words[at] = keep.raw();
        }

        let mut clone_groups = std::collections::BTreeMap::<u32, Vec<(usize, usize)>>::new();
        for (variable, pointer, format) in variables {
            if requested_by_variable.get(&variable).copied() == Some(format) && format != original {
                changed += 1;
            }
            if format != keep {
                clone_groups
                    .entry(format.raw())
                    .or_default()
                    .push((variable, pointer));
            }
        }
        for (format_raw, group) in clone_groups {
            let (image_start, image_len) =
                image_instruction[image_type].ok_or(ImageFormatSpecializeError::MalformedModule)?;
            let new_image = next_id;
            next_id += 1;
            let mut image_words = words[image_start..image_start + image_len].to_vec();
            image_words[1] = new_image;
            image_words[8] = format_raw;
            extra.extend(image_words);

            let mut pointer_clones = std::collections::BTreeMap::<usize, u32>::new();
            for &(_, pointer) in &group {
                if pointer_clones.contains_key(&pointer) {
                    continue;
                }
                let (pointer_start, pointer_len) = pointer_instruction[pointer]
                    .ok_or(ImageFormatSpecializeError::MalformedModule)?;
                let new_pointer = next_id;
                next_id += 1;
                let mut pointer_words = words[pointer_start..pointer_start + pointer_len].to_vec();
                pointer_words[1] = new_pointer;
                pointer_words[3] = new_image;
                extra.extend(pointer_words);
                pointer_clones.insert(pointer, new_pointer);
            }
            for (variable, pointer) in group {
                let type_word = variable_type_word[variable]
                    .ok_or(ImageFormatSpecializeError::MalformedModule)?;
                words[type_word] = pointer_clones[&pointer];
                retyped.insert(variable as u32, new_image);
            }
        }
    }
    if !extra.is_empty() {
        let at = insert_at.unwrap_or(words.len());
        words.splice(at..at, extra);
        words[3] = next_id;
    }
    // A variable that moved to a cloned image type takes its loads with it. An
    // `OpLoad`'s result type must be the pointee of its pointer, so leaving the
    // loads alone is what makes the module invalid rather than merely
    // differently typed — and this pass is why the clone path is usable at all.
    retype_loads(words, &retyped);
    verify_load_types(words)?;
    for &(binding, format) in requested {
        if image_format(words, binding) != Some(format) {
            return Err(ImageFormatSpecializeError::MissingBinding(binding));
        }
    }
    Ok(changed)
}

/// Point every `OpLoad` of a retyped variable at that variable's new type.
///
/// `retyped` maps a global `OpVariable` id to the `OpTypeImage` its pointer now
/// names. Nothing else in the module refers to the *variable's* type by id, so
/// the loads are the whole repair — `OpImageWrite` and `OpImageRead` take the
/// loaded object and declare no type of their own.
fn retype_loads(words: &mut [u32], retyped: &std::collections::BTreeMap<u32, u32>) {
    if retyped.is_empty() {
        return;
    }
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        if word_count == 0 || i + word_count > words.len() {
            return;
        }
        if (words[i] & 0xffff) as u16 == OP_LOAD && word_count >= 4 {
            if let Some(&image) = retyped.get(&words[i + 3]) {
                words[i + 1] = image;
            }
        }
        i += word_count;
    }
}

/// Refuse a module whose loads and pointees disagree.
///
/// The SPIR-V rule is that an `OpLoad`'s result type is the type its pointer
/// points at. Checked here rather than left to the driver because the driver
/// that finds it may not survive it: an NVIDIA SPIR-V compiler
/// segmentation-faults inside `vkCreateComputePipelines` on this defect and
/// takes the VM process with it. Only loads through a global `OpVariable` are
/// checked, which is every load this function's clone path can have moved.
fn verify_load_types(words: &[u32]) -> Result<(), ImageFormatSpecializeError> {
    let bound = *words
        .get(3)
        .ok_or(ImageFormatSpecializeError::MalformedModule)? as usize;
    let mut pointer_pointee = vec![None; bound];
    let mut variable_type = vec![None; bound];
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = (words[i] & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            return Err(ImageFormatSpecializeError::MalformedModule);
        }
        match opcode {
            OP_TYPE_POINTER if word_count >= 4 => {
                let id = words[i + 1] as usize;
                if id < bound {
                    pointer_pointee[id] = Some(words[i + 3]);
                }
            }
            OP_VARIABLE if word_count >= 4 => {
                let id = words[i + 2] as usize;
                if id < bound {
                    variable_type[id] = Some(words[i + 1] as usize);
                }
            }
            OP_LOAD if word_count >= 4 => {
                let pointer = words[i + 3] as usize;
                let pointee = variable_type
                    .get(pointer)
                    .copied()
                    .flatten()
                    .and_then(|p| pointer_pointee.get(p).copied().flatten());
                if let Some(pointee) = pointee {
                    if pointee != words[i + 1] {
                        return Err(ImageFormatSpecializeError::LoadTypeMismatch {
                            pointer: words[i + 3],
                            declared: words[i + 1],
                        });
                    }
                }
            }
            _ => {}
        }
        i += word_count;
    }
    Ok(())
}

/// Ensure the module declares `OpCapability StorageImageWriteWithoutFormat`.
///
/// A storage image whose `OpTypeImage` format is `Unknown` (SPIR-V value 0) may
/// only be written when the module declares this capability (and the device
/// enables the matching Vulkan feature). The device retargets a guest
/// `BGRA8Unorm` storage surface to an `Unknown`-format image viewed
/// `B8G8R8A8_UNORM`, so it must add the capability to the translated kernel,
/// which declares only `Shader`/`Float16`/… Idempotent: does nothing if already
/// present. Capabilities occupy the module's first section (right after the
/// 5-word header), so the new instruction is spliced immediately after the last
/// existing `OpCapability`. Returns `true` if it inserted the capability.
pub fn ensure_storage_write_without_format_capability(words: &mut Vec<u32>) -> bool {
    ensure_capability(words, CAPABILITY_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT)
}

/// Splice `OpCapability <cap>` into a module that does not already declare it.
///
/// Capabilities occupy the module's first section, immediately after the 5-word
/// header, so the instruction goes after the last existing `OpCapability`.
/// Idempotent; returns `true` if it inserted one.
fn ensure_capability(words: &mut Vec<u32>, cap: u32) -> bool {
    if words.len() < HEADER_WORDS {
        return false;
    }
    let mut i = HEADER_WORDS;
    let mut insert_at = HEADER_WORDS;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = (words[i] & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        if opcode != OP_CAPABILITY {
            break;
        }
        if word_count >= 2 && words[i + 1] == cap {
            return false;
        }
        i += word_count;
        insert_at = i;
    }
    let instr = [(2u32 << 16) | OP_CAPABILITY as u32, cap];
    words.splice(insert_at..insert_at, instr);
    true
}

/// The storage-image capabilities a finished module's own contents require.
///
/// Every field is `true` only if some instruction in the module cannot be
/// validated without the matching `OpCapability`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RequiredImageCapabilities {
    /// A storage image declares a format outside SPIR-V's core set.
    pub extended_formats: bool,
    /// An `OpImageWrite` targets a storage image whose format is `Unknown`.
    pub write_without_format: bool,
    /// An `OpImageRead` targets a storage image whose format is `Unknown`.
    pub read_without_format: bool,
}

impl RequiredImageCapabilities {
    pub fn any(&self) -> bool {
        self.extended_formats || self.write_without_format || self.read_without_format
    }
}

/// Whether a SPIR-V storage-image format needs `StorageImageExtendedFormats`.
///
/// The core set — usable with `Shader` alone — is `Unknown`, the four `Rgba*32/16`
/// and `R32` float/int/uint widths, `Rgba8`/`Rgba8Snorm`, and the 8-bit `Rgba8i`
/// / `Rgba8ui` forms. Anything narrower than four channels below 32 bits, which
/// is where `Rg16f`, `R16f`, `Rg8` and `R8` all live, is extended.
///
/// Written as an explicit list of the core values rather than a range, because
/// the core set is not contiguous in the enum and a `<=` test would silently
/// admit the two-channel formats sitting between the four-channel ones.
fn storage_format_is_extended(raw: u32) -> bool {
    !matches!(
        raw,
        0     // Unknown
        | 1   // Rgba32f
        | 2   // Rgba16f
        | 3   // R32f
        | 4   // Rgba8
        | 5   // Rgba8Snorm
        | 21  // Rgba32i
        | 22  // Rgba16i
        | 23  // Rgba8i
        | 24  // R32i
        | 30  // Rgba32ui
        | 31  // Rgba16ui
        | 32  // Rgba8ui
        | 33 // R32ui
    )
}

/// Derive which storage-image capabilities a module needs, from the module.
///
/// # Why this is derived and not tracked
///
/// The device used to add `StorageImageWriteWithoutFormat` when *it* had
/// retargeted a binding to `Unknown`, on the reasoning that this was the only
/// way a module could come to need it. That is a claim about provenance, and it
/// was wrong in the direction that reads as careful: the translator emits
/// `Unknown`-format storage images of its own accord, and those modules reached
/// the validator without the capability and were **rejected**, losing the
/// dispatch. Both x86 rails measured here lose compute work to exactly that —
/// including the rail that otherwise renders correctly, which is why it went
/// unnoticed.
///
/// Asking the module what it contains cannot go stale that way. A new source of
/// extended-format or format-less storage images — another translator version, a
/// new specialization, a guest that simply emits one — is covered without anyone
/// remembering to add a case, which is the property the provenance test did not
/// have.
///
/// # Why it over-approximates on purpose
///
/// The format requirement is exact: a storage image's declared format is right
/// there in its `OpTypeImage`. The *format-less* requirement is not attributed
/// to a specific write. This walk asks "does the module declare an
/// `Unknown`-format storage image, and does it contain any `OpImageWrite`",
/// rather than proving that some particular write targets that image.
///
/// That is deliberate, and the first version of this function did it the other
/// way. Resolving the write's image operand back through
/// `OpTypePointer`/`OpVariable`/`OpLoad` attributed most writes correctly and
/// **missed three modules per boot** — the def-use chain reached them through a
/// shape the walk did not follow, and the only symptom was the same rejection
/// the function existed to prevent. Every shape it misses fails closed in the
/// direction that loses guest work, and there is no bound on how many shapes a
/// translator can emit.
///
/// Over-approximating cannot fail that way, and it costs nothing real:
/// declaring a capability a module does not use is valid SPIR-V, and the caller
/// has already established that the device enabled the matching feature. The
/// exchange is a capability that is occasionally redundant for a class of bug
/// that is silent and recurring.
///
/// The walk is structural throughout and never looks at debug names or guest
/// object ids.
pub fn required_image_capabilities(words: &[u32]) -> RequiredImageCapabilities {
    let mut need = RequiredImageCapabilities::default();
    if words.len() < HEADER_WORDS {
        return need;
    }
    let mut has_unknown_storage_image = false;
    let mut has_image_write = false;
    let mut has_image_read = false;

    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = (words[i] & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        match opcode {
            OP_TYPE_IMAGE if word_count >= 9 => {
                // [7] = Sampled, [8] = Image Format. Sampled 0 means "either",
                // so it is a possible storage use and counts.
                let sampled = words[i + 7];
                let format = words[i + 8];
                if sampled == IMAGE_SAMPLED_STORAGE || sampled == 0 {
                    if storage_format_is_extended(format) {
                        need.extended_formats = true;
                    }
                    if format == 0 {
                        has_unknown_storage_image = true;
                    }
                }
            }
            OP_IMAGE_WRITE => has_image_write = true,
            OP_IMAGE_READ => has_image_read = true,
            _ => {}
        }
        i += word_count;
    }
    need.write_without_format = has_unknown_storage_image && has_image_write;
    need.read_without_format = has_unknown_storage_image && has_image_read;
    need
}

/// Every `OpTypeImage` in a module, as `(sampled, format)` pairs.
///
/// Diagnostic only. When the validator refuses a module for a capability the
/// derivation above did not ask for, the disagreement is between what
/// `spirv-val` sees and what this walk sees, and the only way to settle it is to
/// print what the walk found. Ordered as encountered, deduplicated.
pub fn image_type_census(words: &[u32]) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    if words.len() < HEADER_WORDS {
        return out;
    }
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = (words[i] & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        if opcode == OP_TYPE_IMAGE && word_count >= 9 {
            let pair = (words[i + 7], words[i + 8]);
            if !out.contains(&pair) {
                out.push(pair);
            }
        }
        i += word_count;
    }
    out
}

/// Add every capability [`required_image_capabilities`] found missing.
///
/// Returns what it added. Declaring a capability whose Vulkan feature the device
/// did not enable is invalid usage, so the caller must gate on the features and
/// decline by name rather than hand the driver a module it may not survive.
pub fn ensure_image_capabilities(
    words: &mut Vec<u32>,
    need: &RequiredImageCapabilities,
) -> RequiredImageCapabilities {
    let mut added = RequiredImageCapabilities::default();
    if need.extended_formats {
        added.extended_formats =
            ensure_capability(words, CAPABILITY_STORAGE_IMAGE_EXTENDED_FORMATS);
    }
    if need.write_without_format {
        added.write_without_format =
            ensure_capability(words, CAPABILITY_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT);
    }
    if need.read_without_format {
        added.read_without_format =
            ensure_capability(words, CAPABILITY_STORAGE_IMAGE_READ_WITHOUT_FORMAT);
    }
    added
}

/// Reflect every set-0 sampler descriptor binding declared by a SPIR-V module.
///
/// This includes separate `OpTypeSampler` descriptors used by explicit and AIR
/// static samplers, plus combined `OpTypeSampledImage` descriptors. The walk is
/// structural and does not depend on debug names or guest object identifiers.
pub fn sampler_bindings(words: &[u32]) -> Vec<u32> {
    use std::collections::HashSet;

    let mut sampler_types = HashSet::new();
    let mut sampler_ptrs = HashSet::new();
    let mut sampler_vars = HashSet::new();
    let mut decorations = Vec::new();
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word0 = words[i];
        let word_count = (word0 >> 16) as usize;
        let opcode = (word0 & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        match opcode {
            OP_TYPE_SAMPLER | OP_TYPE_SAMPLED_IMAGE if word_count >= 2 => {
                sampler_types.insert(words[i + 1]);
            }
            OP_TYPE_POINTER if word_count >= 4 => {
                if words[i + 2] == STORAGE_CLASS_UNIFORM_CONSTANT
                    && sampler_types.contains(&words[i + 3])
                {
                    sampler_ptrs.insert(words[i + 1]);
                }
            }
            OP_VARIABLE if word_count >= 4 => {
                if sampler_ptrs.contains(&words[i + 1])
                    && words[i + 3] == STORAGE_CLASS_UNIFORM_CONSTANT
                {
                    sampler_vars.insert(words[i + 2]);
                }
            }
            OP_DECORATE if word_count >= 4 && words[i + 2] == DECORATION_BINDING => {
                decorations.push((words[i + 1], words[i + 3]));
            }
            _ => {}
        }
        i += word_count;
    }
    let mut bindings: Vec<u32> = decorations
        .into_iter()
        .filter_map(|(id, binding)| sampler_vars.contains(&id).then_some(binding))
        .collect();
    bindings.sort_unstable();
    bindings.dedup();
    bindings
}

/// SPIR-V `Dim` operand value `SubpassData` — the framebuffer-fetch image.
const DIM_SUBPASS_DATA: u32 = 6;

/// Which Metal argument class a set-0 descriptor variable was translated from.
///
/// metal2vulkan numbers each class from its own base — [`TEXTURE_BINDING_BASE`]
/// `+ N` for texture `N`, [`SAMPLER_BINDING_BASE`] `+ N` for sampler `N` — and
/// those bases sit 32 apart. So a binding number names its class only while
/// every Metal index stays under 32: texture 40 and sampler 8 are both binding
/// 72. The class is therefore read from the SPIR-V *type* behind the variable,
/// which is exact and stays exact however wide the index gets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindingClass {
    /// `[[texture(n)]]` — an `OpTypeImage` that is not `SubpassData`, sampled or
    /// storage alike.
    Texture,
    /// `[[sampler(n)]]`, including an AIR constexpr sampler. `OpTypeSampler`,
    /// and `OpTypeSampledImage` for the combined form.
    Sampler,
    /// `[[color(n)]]` framebuffer fetch: an `OpTypeImage` whose `Dim` is
    /// `SubpassData`. Held apart from [`Self::Texture`] because the engine binds
    /// the input attachment at its un-relocated number.
    ColorInput,
    /// A descriptor whose type resolved to none of the above — a `[[buffer(n)]]`
    /// uniform or storage buffer.
    Buffer,
}

/// Resolve every SPIR-V id in `words` to the [`BindingClass`] of the type behind
/// it, indexed by id.
///
/// `None` at an id means either "not a descriptor variable" or "the
/// variable → pointer → pointee chain did not resolve", and the two are the same
/// answer to the only question asked here: this walk cannot name that variable's
/// class. Callers must not treat `None` as a class.
fn variable_classes(words: &[u32]) -> Vec<Option<BindingClass>> {
    let Some(&bound) = words.get(3) else {
        return Vec::new();
    };
    let bound = bound as usize;
    if words.len() < HEADER_WORDS || bound == 0 {
        return Vec::new();
    }
    // Class of a *type* id, the pointee a descriptor pointer names.
    let mut type_class: Vec<Option<BindingClass>> = vec![None; bound];
    let mut pointee: Vec<Option<usize>> = vec![None; bound];
    let mut var_type: Vec<Option<usize>> = vec![None; bound];

    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word0 = words[i];
        let word_count = (word0 >> 16) as usize;
        let opcode = (word0 & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        let mut set_type = |id: u32, class: BindingClass| {
            if (id as usize) < bound {
                type_class[id as usize] = Some(class);
            }
        };
        match opcode {
            // Dim is the third operand; SubpassData is the framebuffer-fetch
            // image and everything else is a Metal texture.
            OP_TYPE_IMAGE if word_count >= 9 => {
                let class = if words[i + 3] == DIM_SUBPASS_DATA {
                    BindingClass::ColorInput
                } else {
                    BindingClass::Texture
                };
                set_type(words[i + 1], class);
            }
            OP_TYPE_SAMPLER | OP_TYPE_SAMPLED_IMAGE if word_count >= 2 => {
                set_type(words[i + 1], BindingClass::Sampler);
            }
            OP_TYPE_POINTER if word_count >= 4 => {
                let id = words[i + 1] as usize;
                if id < bound {
                    pointee[id] = Some(words[i + 3] as usize);
                }
            }
            OP_VARIABLE if word_count >= 4 => {
                let id = words[i + 2] as usize;
                if id < bound {
                    var_type[id] = Some(words[i + 1] as usize);
                }
            }
            _ => {}
        }
        i += word_count;
    }

    (0..bound)
        .map(|id| {
            let pointer = var_type[id]?;
            let pointee = pointee.get(pointer).copied().flatten()?;
            // A resolved chain that named no image or sampler type is a buffer;
            // an unresolved one is `None` and stays unnamed.
            Some(
                type_class
                    .get(pointee)
                    .copied()
                    .flatten()
                    .unwrap_or(BindingClass::Buffer),
            )
        })
        .collect()
}

/// A `Binding` decoration whose variable [`variable_classes`] could not name, so
/// the relocation fell back to the binding number's band.
///
/// The fallback is only correct while every Metal index is under 32, which is
/// what makes this worth reporting rather than absorbing: it is the one input
/// shape that would make a widened band mis-relocate.
struct UnclassifiedBinding {
    binding: u32,
    variable: u32,
}

impl crate::observe::Decline for UnclassifiedBinding {
    fn slug(&self) -> &'static str {
        "spirv_reloc_unclassified_binding"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("binding", self.binding.to_string()),
            ("variable", self.variable.to_string()),
        ]
    }
}

/// Add `offset` to every `Binding` decoration whose variable is in `classes`.
///
/// The class comes from the variable's SPIR-V type ([`variable_classes`]); the
/// `band` predicate is consulted only for a variable that walk could not name,
/// and each such fallback is reported. Both rules agree for every Metal index
/// under 32 — the type rule is the one that keeps agreeing above it.
fn relocate_by_class(
    words: &mut [u32],
    classes: &[BindingClass],
    band: impl Fn(u32) -> bool,
    offset: u32,
) -> usize {
    let by_variable = variable_classes(words);
    let mut rewritten = 0usize;
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word0 = words[i];
        let word_count = (word0 >> 16) as usize;
        let opcode = (word0 & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        if opcode == OP_DECORATE && word_count >= 4 && words[i + 2] == DECORATION_BINDING {
            let variable = words[i + 1];
            let binding = words[i + 3];
            let wanted = match by_variable.get(variable as usize).copied().flatten() {
                Some(class) => classes.contains(&class),
                None => {
                    crate::observe::Emit::decline(
                        "spirv_reloc",
                        &UnclassifiedBinding { binding, variable },
                    )
                    .fail_once(u64::from(binding));
                    band(binding)
                }
            };
            if wanted {
                words[i + 3] = binding + offset;
                rewritten += 1;
            }
        }
        i += word_count;
    }
    rewritten
}

/// Rewrite fragment SPIR-V: buffer bindings += [`FRAG_BUFFER_BINDING_OFFSET`]
/// (source band `[0,32)`, destination `[104,136)`, clear of the `[96,104)`
/// ColorInput band).
///
/// # Neither relocation ran on a driven x86/PCI boot
///
/// `m2v_cache::fragment_words` returns the unrelocated words when both
/// `separate_sampled` and `buf_collide` are false, and on a driven boot
/// (web-content probe, 10 captures, 494 draws in a census window) that was every
/// shader: 160 `linux_m2v_async` lines and **zero** `frag_sampled_reloc` or
/// `frag_buf_reloc` lines. Both are `observe::line` on the same channel and gate
/// as the `linux_m2v_async` lines that did appear, so the absence is a reading
/// and not a suppressed sink.
///
/// So this guest's WindowServer/Safari compositing does not put textures in both
/// stages, nor collide a buffer index across them. That is what the relocation
/// exists for, and it is a real Metal shape rather than a dead arm — but it
/// means a boot on this workload cannot regression-test either function, and the
/// unit tests are the only coverage. Do not read a green boot as exercising
/// them.
pub fn offset_fragment_buffer_bindings(words: &mut [u32]) -> usize {
    relocate_by_class(
        words,
        &[BindingClass::Buffer],
        |binding| binding < SAMPLED_RESOURCE_BINDING_BASE,
        FRAG_BUFFER_BINDING_OFFSET,
    )
}

/// Rewrite a freshly translated module from the translator's narrow bands into
/// the device's wide ones: sampler and ColorInput bindings +=
/// [`SAMPLED_TAIL_WIDEN_OFFSET`]. Textures do not move.
///
/// Run once per shader, before anything reads a binding number and before either
/// fragment relocation, so every consumer downstream sees one numbering.
///
/// # This is what makes texture indices 32..127 reachable
///
/// The translator's bands are 32 apart, so it decorates Metal texture 40 with
/// binding 72 — the number it also gives sampler 8. Every index at or above 32
/// was therefore refused upstream, and Apple's serializer emits up to 128
/// (`bind_limit::TEXTURE`). Moving the two tail bands up leaves the texture band
/// 128 wide, and the texture decorations are already correct in it.
///
/// # Why it cannot mis-file a binding
///
/// The class comes from the variable's SPIR-V type, not its number
/// ([`variable_classes`]): a texture is an `OpTypeImage` whose `Dim` is not
/// `SubpassData`, a sampler an `OpTypeSampler`, a ColorInput a `SubpassData`
/// image. So the pass separates a texture at 72 from a sampler at 72 exactly.
/// It also *repairs* a module in which the translator gave both the same number
/// — two variables that arrived colliding leave as two — which is the one shape
/// the narrow bands could not express at all.
///
/// The band predicate is the fallback for a variable the type walk could not
/// name, and it is only consulted then; each such fallback is reported as
/// `spirv_reloc_unclassified_binding`. Three driven boots covering 160 shader
/// translations each report none, so the type rule names every variable in every
/// shader this guest ships.
///
/// # What a driven boot says about it, and what it cannot
///
/// Driven x86/PCI boot after this pass landed (web-content probe, 10 captures):
/// 10 of 10 regions measured their declared colour, `spirv_reloc_unclassified_binding`
/// and `m2v_reflect_malformed` both absent, and the fail-channel reason ranking
/// unchanged in shape from the boot before it. That is a regression check on the
/// relocation over 160 real guest shaders — it says the widened numbering did not
/// break the shaders that already worked.
///
/// It is **not** evidence about the widening itself. The same boot reads every
/// one of its bind records in the `le16` band: `render_bind_reach_texture_le16`
/// = 9 290, with `le_table` and `over_table` both absent. This guest's
/// WindowServer/Safari compositing never binds a texture above slot 16, so no
/// boot on this workload can exercise slots 32..127 at all. The coverage for
/// those is `exec::tests::a_texture_bind_past_the_old_band_binds_and_keeps_its_own_descriptor`
/// and the `const` assertions on the band map.
pub fn widen_sampled_bands(words: &mut [u32]) -> usize {
    relocate_by_class(
        words,
        &[BindingClass::Sampler, BindingClass::ColorInput],
        |binding| binding >= M2V_SAMPLER_BINDING_BASE,
        SAMPLED_TAIL_WIDEN_OFFSET,
    )
}

/// Rewrite fragment SPIR-V: texture and sampler bindings +=
/// [`FRAG_SAMPLED_RESOURCE_BINDING_OFFSET`].
///
/// The source band is the device's, not the translator's: textures `[32,160)`
/// plus samplers `[160,192)`, i.e. everything below
/// [`SAMPLED_RESOURCE_BINDING_LIMIT`]. This runs after [`widen_sampled_bands`],
/// so a sampler here is already at 160+N.
///
/// The ColorInput band ([`COLOR_INPUT_BINDING_BASE`]) is deliberately NOT
/// relocated: the engine binds the framebuffer-fetch input attachment by its
/// un-relocated number, exactly like the storage/descriptor reflectors key on
/// un-relocated bindings. That exclusion is now the image's `SubpassData` `Dim`
/// rather than its binding number, so it holds for a texture index that reaches
/// the band numerically.
pub fn offset_fragment_sampled_resource_bindings(words: &mut [u32]) -> usize {
    relocate_by_class(
        words,
        &[BindingClass::Texture, BindingClass::Sampler],
        |binding| {
            (SAMPLED_RESOURCE_BINDING_BASE..SAMPLED_RESOURCE_BINDING_LIMIT).contains(&binding)
        },
        FRAG_SAMPLED_RESOURCE_BINDING_OFFSET,
    )
}

// ---------------------------------------------------------------------------
// Reflection-derived reflectors (single source of truth) + divergence census
// ---------------------------------------------------------------------------
//
// `metal2vulkan::reflect::ShaderReflection` already carries the decoded texture
// shape / access per binding, parsed from the AIR by the SAME decoder the emit
// path uses to write the `OpTypeImage`. The functions below read those facts
// directly, so a consumer never re-walks the emitted SPIR-V. They are keyed on
// the descriptor binding EXACTLY as reflection reports it — the UN-relocated
// number (`TEXTURE_BINDING_BASE + metal_index`), before the fragment
// sampled-resource relocation a merged-stage draw later applies. Textures are
// the one class where that number is the same in both numberings, which is why
// these reflectors did not have to move when the bands widened; a sampler
// lookup here is in the translator's `M2V_SAMPLER_BINDING_BASE` band.
//
// The `census_reflection_wellformed` guard runs once per translate (miss path)
// and validates, on the live guest's own shaders, that the AIR-derived reflection
// is internally consistent and ABI-versioned. It is the always-on regression
// proxy for the hot path now that texture shape/access is read solely from
// reflection (no second SPIR-V walk to cross-check against).

use metal2vulkan::meta::{TextureDimension, TextureShape};
use metal2vulkan::reflect::{
    ResourceAccess, ResourceKind, ShaderReflection, ShaderStage, REFLECTION_VERSION,
    RESOURCE_DESCRIPTOR_SET,
};

/// Map a decoded [`TextureShape`] to a [`SampledImageKind`] via its `OpTypeImage`
/// Dim + Arrayed. `None` for shapes `SampledImageKind` cannot express (a texel
/// `Buffer`, or a 3D array) — those are legitimate reflection shapes the sampled
/// render path does not support and rejects fail-visibly at the call site.
fn sampled_image_kind_from_shape(shape: &TextureShape) -> Option<SampledImageKind> {
    match (shape.dimension, shape.arrayed) {
        (TextureDimension::D1, false) => Some(SampledImageKind::D1),
        (TextureDimension::D1, true) => Some(SampledImageKind::D1Array),
        (TextureDimension::D2, false) => Some(SampledImageKind::D2),
        (TextureDimension::D2, true) => Some(SampledImageKind::D2Array),
        (TextureDimension::D3, false) => Some(SampledImageKind::D3),
        (TextureDimension::Cube, false) => Some(SampledImageKind::Cube),
        (TextureDimension::Cube, true) => Some(SampledImageKind::CubeArray),
        _ => None,
    }
}

/// Find the texture shape reflection reports for descriptor `binding` (the
/// UN-relocated number). `None` when no binding matches or it carries no shape.
fn texture_shape_for_binding(reflection: &ShaderReflection, binding: u32) -> Option<&TextureShape> {
    reflection.bindings.iter().find_map(|b| {
        (is_texture_kind(b.kind) && b.descriptor.map(|d| d.binding) == Some(binding))
            .then_some(b.texture_shape.as_ref())
            .flatten()
    })
}

/// Whether a reflected resource is one of the kinds a `[[texture(n)]]` index
/// names, as opposed to a sampler, a buffer, or a framebuffer-fetch input.
///
/// The kind is checked as well as the binding because reflection reports the
/// *translator's* numbering, whose bands are 32 apart: Metal texture 64 and
/// ColorInput 0 are both binding 96 there, and both carry a `texture_shape`. The
/// SPIR-V has no such ambiguity — [`widen_sampled_bands`] separated the two —
/// but a lookup into reflection still has to say which one it meant, and the
/// kind is the field that says it.
fn is_texture_kind(kind: ResourceKind) -> bool {
    matches!(
        kind,
        ResourceKind::Texture
            | ResourceKind::TextureArray
            | ResourceKind::StorageImage
            | ResourceKind::EmbeddedArgBufferTexture
    )
}

/// How reflection describes descriptor `binding` for the sampled render path.
/// Lets the call site log a genuine gap fail-visibly while staying silent on the
/// expected "bound but not sampled by this shader" case.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReflectedSampledKind {
    /// Reflection carries a sampled dimensionality the render path can express.
    Kind(SampledImageKind),
    /// Reflection lists a texture shape here the sampled path cannot express (a
    /// texel `Buffer` or a 3D array) — a genuine unsupported shape.
    Unsupported,
    /// Reflection lists no texture shape at this binding — an unused/unbound slot
    /// (Metal permits binding a texture a shader never samples).
    Absent,
}

/// Classify descriptor `binding` for the sampled render path from reflection.
pub fn reflected_sampled_kind(reflection: &ShaderReflection, binding: u32) -> ReflectedSampledKind {
    match texture_shape_for_binding(reflection, binding) {
        None => ReflectedSampledKind::Absent,
        Some(shape) => match sampled_image_kind_from_shape(shape) {
            Some(kind) => ReflectedSampledKind::Kind(kind),
            None => ReflectedSampledKind::Unsupported,
        },
    }
}

/// How the compute rail must treat texture descriptor `binding`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReflectedComputeTexture {
    /// Reflection lists no texture shape here. Metal permits binding a texture
    /// the shader never samples or writes, so this is expected control flow —
    /// the caller stages nothing and invents no access semantics for it.
    Absent,
    /// A single-layer, non-multisampled 2D texture, carrying its
    /// sampled-vs-storage class. This is the only shape the compute rail can
    /// stage: a binding comes from one type-11 plane window or one linear GVA
    /// level, both flat `width × height` rectangles.
    Plain2d(ImageAccess),
    /// The shader declares a shape with a slice, depth, or sample axis the
    /// compute rail has no staged source for. `axis` names it for the fail log.
    UnstageableShape { axis: &'static str },
}

/// Classify texture descriptor `binding` for the compute rail from the
/// translator's reflection.
///
/// Sampled-vs-storage comes from the declared Metal access qualifier
/// (`TextureShape.writable`), which is exact at translate time — there is no
/// `Unknown`. The shape axis comes from the same decoded `OpTypeImage`, and
/// the rail refuses anything it would otherwise stage as 2D behind the
/// shader's back: binding a `TYPE_2D` view to a SPIR-V image declared
/// `2DArray`/`3D`/`1D`/`Cube`/`Buffer`/multisampled is a descriptor-type
/// mismatch, not a degraded render.
pub fn reflected_compute_texture(
    reflection: &ShaderReflection,
    binding: u32,
) -> ReflectedComputeTexture {
    let Some(shape) = texture_shape_for_binding(reflection, binding) else {
        return ReflectedComputeTexture::Absent;
    };
    let axis = match shape.dimension {
        TextureDimension::D1 => Some("dim_1d"),
        TextureDimension::D3 => Some("dim_3d"),
        TextureDimension::Cube => Some("dim_cube"),
        TextureDimension::Buffer => Some("dim_buffer"),
        TextureDimension::D2 if shape.arrayed => Some("arrayed"),
        TextureDimension::D2 if shape.multisampled => Some("multisampled"),
        TextureDimension::D2 => None,
    };
    match axis {
        Some(axis) => ReflectedComputeTexture::UnstageableShape { axis },
        None => ReflectedComputeTexture::Plain2d(if shape.writable {
            ImageAccess::Storage
        } else {
            ImageAccess::Sampled
        }),
    }
}

/// Validate that the translator's reflection is internally well-formed, once per
/// translate (miss path). This is the always-on regression proxy that replaces
/// the former reflection-vs-SPIR-V cross-check: the hot path now reads texture
/// shape and sampled-vs-storage access solely from reflection, so the guard must
/// catch a reflection that is self-contradictory or emitted against a different
/// ABI — without a second walk of the SPIR-V. Checks:
///   - `reflection_version` matches the ABI this consumer was built against
///     (catches a translator/consumer version skew);
///   - each static sampler carries decoded state and a set-0 descriptor inside
///     the sampler ABI band;
///   - each texture-family binding carries a descriptor location;
///   - the redundant sampled-vs-storage encodings agree — `ResourceKind`
///     (`StorageImage`), `TextureShape.writable`, and `ResourceAccess`. The
///     translator derives all three from one decoded `TextureShape`, so any
///     disagreement is a translator regression the consumer must not trust.
///
/// Logs `m2v_reflect_malformed reason=<slug>` fail-visibly; quiet on a healthy
/// boot (returns the number of violations found, 0 when clean).
pub fn census_reflection_wellformed(reflection: &ShaderReflection, pipeline_ref: u32) -> usize {
    let mut bad = 0;
    if reflection.reflection_version != REFLECTION_VERSION {
        bad += 1;
        crate::observe::fail(format!(
            "m2v_reflect_malformed pipe={pipeline_ref} reason=reflection_version_mismatch \
             got={} want={REFLECTION_VERSION}",
            reflection.reflection_version
        ));
    }
    for b in &reflection.bindings {
        if b.kind == ResourceKind::StaticSampler {
            match (b.descriptor, b.static_sampler) {
                (None, _) => {
                    bad += 1;
                    crate::observe::fail(format!(
                        "m2v_reflect_malformed pipe={pipeline_ref} \
                         reason=static_sampler_no_descriptor metal_index={}",
                        b.metal_index
                    ));
                }
                (Some(descriptor), None) => {
                    bad += 1;
                    crate::observe::fail(format!(
                        "m2v_reflect_malformed pipe={pipeline_ref} \
                         reason=static_sampler_no_state bind={}",
                        descriptor.binding
                    ));
                }
                (Some(descriptor), Some(_))
                    if descriptor.set != RESOURCE_DESCRIPTOR_SET
                        || !(M2V_SAMPLER_BINDING_BASE..M2V_COLOR_INPUT_BINDING_BASE)
                            .contains(&descriptor.binding) =>
                {
                    bad += 1;
                    crate::observe::fail(format!(
                        "m2v_reflect_malformed pipe={pipeline_ref} \
                         reason=static_sampler_descriptor_out_of_band set={} bind={} \
                         expected_set={RESOURCE_DESCRIPTOR_SET} expected_band={}..{}",
                        descriptor.set,
                        descriptor.binding,
                        M2V_SAMPLER_BINDING_BASE,
                        M2V_COLOR_INPUT_BINDING_BASE
                    ));
                }
                (Some(_), Some(_)) => {}
            }
            continue;
        }
        if b.static_sampler.is_some() {
            bad += 1;
            crate::observe::fail(format!(
                "m2v_reflect_malformed pipe={pipeline_ref} \
                 reason=static_sampler_state_on_nonstatic kind={:?} metal_index={}",
                b.kind, b.metal_index
            ));
        }
        let texture_family = matches!(
            b.kind,
            ResourceKind::Texture
                | ResourceKind::TextureArray
                | ResourceKind::StorageImage
                | ResourceKind::EmbeddedArgBufferTexture
        );
        if !texture_family {
            continue;
        }
        let binding = b.descriptor.map(|d| d.binding);
        if binding.is_none() {
            bad += 1;
            crate::observe::fail(format!(
                "m2v_reflect_malformed pipe={pipeline_ref} reason=texture_binding_no_descriptor \
                 kind={:?} metal_index={}",
                b.kind, b.metal_index
            ));
        }
        // Only the two malformed-reflection lines below read this, and a missing
        // descriptor has already emitted its own. Rendering that case as `0`
        // names a real binding index the reflection never carried, so the two
        // failures downstream would read as being about binding 0.
        let bind = match binding {
            Some(bind) => bind.to_string(),
            None => "none".to_string(),
        };
        // Storage-vs-sampled must agree across the three encodings the consumer
        // and the translator both derive from the one `TextureShape`.
        let kind_storage = matches!(b.kind, ResourceKind::StorageImage);
        if let Some(writable) = b.texture_shape.as_ref().map(|s| s.writable) {
            if writable != kind_storage {
                bad += 1;
                crate::observe::fail(format!(
                    "m2v_reflect_malformed pipe={pipeline_ref} reason=kind_writable_disagree \
                     bind={bind} kind={:?} writable={writable}",
                    b.kind
                ));
            }
        }
        let access_storage = match b.access {
            Some(ResourceAccess::Storage) => Some(true),
            Some(ResourceAccess::Sampled) => Some(false),
            _ => None,
        };
        if let Some(access_storage) = access_storage {
            if access_storage != kind_storage {
                bad += 1;
                crate::observe::fail(format!(
                    "m2v_reflect_malformed pipe={pipeline_ref} reason=kind_access_disagree \
                     bind={bind} kind={:?} access={:?}",
                    b.kind, b.access
                ));
            }
        }
    }
    bad
}

/// Report a shader's runtime `[[function_constant(N)]]` inventory when it declares any.
///
/// metal2vulkan does not plumb runtime function-constant specialization: its emit
/// path folds every `[[function_constant]]` load to its disabled default (0) and
/// selects that variant (`passes::transform_with_options` `fold_function_constants`).
/// The paravirt command stream, in turn, carries no `MTLFunctionConstantValues` for
/// us to apply — the pipeline/function descriptors decode only refs + the AIR blob,
/// and that AIR is unspecialized (its `air.fc_initializer` globals are
/// `externally_initialized undef`). So a shader that declares runtime function
/// constants is always translated as its FC-disabled variant.
///
/// That is the current, accepted behavior — it renders the system UI clean — but it
/// is a real gap for any shader whose guest-selected FC values differ from the
/// disabled default. This once-per-translate line makes the reliance MEASURABLE:
/// which shaders (by Metal entry name) carry runtime FCs, so a future rendering
/// delta can be correlated with FC usage and the specialization gap sized before any
/// fix. It is diagnostic, not a per-draw failure, so it goes to the OFF-prefixed
/// analysis sink (not `fail`, which must read zero on a healthy boot).
///
/// The input is the reflection's `function_constants` — the translator's single
/// source of truth, scanned once from the AIR `air.fc_initializer` ABI globals — so
/// there is no SPIR-V re-walk. Silent for the common FC-free shader. Returns the
/// count reported (0 = silent) for tests.
pub fn log_folded_function_constants(reflection: &ShaderReflection) -> usize {
    if reflection.function_constants.is_empty() {
        return 0;
    }
    let stage = match reflection.stage {
        ShaderStage::Vertex => "v",
        ShaderStage::Fragment => "f",
        ShaderStage::Kernel => "k",
    };
    let entry = reflection.entry_point.as_deref().unwrap_or("?");
    let inventory: Vec<String> = reflection
        .function_constants
        .iter()
        .map(|fc| format!("{}:{}:{}", fc.index, fc.name, fc.type_name))
        .collect();
    crate::observe::off(format!(
        "fc_folded_disabled stage={stage} entry={entry} count={} fcs=[{}]",
        inventory.len(),
        inventory.join(",")
    ));
    reflection.function_constants.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injects_storage_write_without_format_capability_once() {
        // 5-word header, `OpCapability Shader`, then `OpMemoryModel` (opcode 14).
        let mut words = vec![
            0x0723_0203,       // magic
            0x0001_0600,       // version
            0,                 // generator
            16,                // bound
            0,                 // schema
            (2u32 << 16) | 17, // OpCapability ...
            1,                 // Shader
            (3u32 << 16) | 14, // OpMemoryModel ...
            0,                 // Logical
            1,                 // GLSL450
        ];
        let before = words.len();
        assert!(ensure_storage_write_without_format_capability(&mut words));
        assert_eq!(words.len(), before + 2);
        // The original Shader capability is untouched at the front.
        assert_eq!(words[5], (2u32 << 16) | 17);
        assert_eq!(words[6], 1);
        // The new capability is spliced after the last OpCapability, before the
        // OpMemoryModel section (capabilities must precede everything else).
        assert_eq!(words[7], (2u32 << 16) | 17);
        // Spelled literally, not as the constant. `assert_eq!(x, X)` where the
        // splice wrote `X` proves the splice ran and nothing about *what it
        // wrote*, and that is precisely how this word sat at 34
        // (`ImageCubeArray`) through every run of this test. The number is
        // SPIR-V's, so the literal is the only side of the comparison that
        // carries information.
        assert_eq!(words[8], 56, "SPIR-V Capability StorageImageWriteWithoutFormat");
        assert_eq!(
            words[9] & 0xffff,
            14,
            "OpMemoryModel follows the capabilities"
        );
        // Idempotent: a second call is a no-op.
        assert!(!ensure_storage_write_without_format_capability(&mut words));
        assert_eq!(words.len(), before + 2);
    }

    /// Build a minimal module: header, `OpCapability Shader`, `OpMemoryModel`,
    /// then whatever body words are given.
    fn module_with(body: &[u32]) -> Vec<u32> {
        let mut words = vec![
            0x0723_0203,       // magic
            0x0001_0600,       // version
            0,                 // generator
            64,                // bound
            0,                 // schema
            (2u32 << 16) | 17, // OpCapability
            1,                 // Shader
            (3u32 << 16) | 14, // OpMemoryModel
            0,
            1,
        ];
        words.extend_from_slice(body);
        words
    }

    /// `OpTypeImage %result %sampled_ty 2D 0 0 0 <sampled> <format>` (9 words).
    fn op_type_image(result: u32, sampled: u32, format: u32) -> [u32; 9] {
        [
            (9u32 << 16) | OP_TYPE_IMAGE as u32,
            result,
            2, // sampled type id (unused by the walk)
            1, // Dim2D
            0, // depth
            0, // arrayed
            0, // MS
            sampled,
            format,
        ]
    }

    /// A storage image declaring an extended format needs the capability, and
    /// this is the exact shape a macOS 11 guest's kernel was rejected for:
    /// `%1138 = OpTypeImage %float 2D 0 0 0 2 Rg16f`.
    #[test]
    fn an_extended_storage_format_requires_its_capability() {
        let words = module_with(&op_type_image(10, IMAGE_SAMPLED_STORAGE, 7));
        let need = required_image_capabilities(&words);
        assert!(need.extended_formats, "Rg16f is outside the core set");
        assert!(!need.write_without_format);
        assert!(!need.read_without_format);
    }

    /// Every core storage format is admitted without a capability.
    ///
    /// Swept rather than sampled because the core set is **not contiguous** in
    /// the enum — the two- and one-channel formats sit between the four-channel
    /// ones — so a `<=` style bound reads as thorough and would quietly demand
    /// the capability for half of them, or waive it for formats that need it.
    #[test]
    fn the_core_storage_formats_need_no_capability() {
        for raw in [0u32, 1, 2, 3, 4, 5, 21, 22, 23, 24, 30, 31, 32, 33] {
            assert!(
                !storage_format_is_extended(raw),
                "format {raw} is in SPIR-V's core storage set"
            );
        }
        // A representative span of the extended ones, including the neighbours
        // of core values, which is where an off-by-one would hide.
        for raw in [6u32, 7, 8, 9, 13, 15, 20, 25, 29, 34, 39] {
            assert!(
                storage_format_is_extended(raw),
                "format {raw} needs StorageImageExtendedFormats"
            );
        }
    }

    /// A *sampled* image is not a storage image, so its format asks for nothing.
    #[test]
    fn a_sampled_image_format_does_not_demand_the_storage_capability() {
        let words = module_with(&op_type_image(10, 1, 7));
        assert!(!required_image_capabilities(&words).extended_formats);
    }

    /// An `Unknown`-format storage image plus a write demands
    /// `StorageImageWriteWithoutFormat`.
    ///
    /// The pointer/variable/load words in the body are the shape the translator
    /// emits around such a write. They are deliberately *not* what the check
    /// depends on — see `required_image_capabilities` on why attributing the
    /// write to its image was tried and abandoned — but they belong in the
    /// fixture so it stays a realistic module rather than two instructions.
    #[test]
    fn a_write_to_an_unknown_format_image_requires_its_capability() {
        let mut body = Vec::new();
        body.extend_from_slice(&op_type_image(10, IMAGE_SAMPLED_STORAGE, 0));
        // OpTypePointer %11 UniformConstant %10
        body.extend_from_slice(&[(4u32 << 16) | OP_TYPE_POINTER as u32, 11, 0, 10]);
        // OpVariable %11 %12 UniformConstant
        body.extend_from_slice(&[(4u32 << 16) | OP_VARIABLE as u32, 11, 12, 0]);
        // OpLoad %10 %13 %12
        body.extend_from_slice(&[(4u32 << 16) | OP_LOAD as u32, 10, 13, 12]);
        // OpImageWrite %13 %20 %21
        body.extend_from_slice(&[(4u32 << 16) | OP_IMAGE_WRITE as u32, 13, 20, 21]);
        let words = module_with(&body);

        let need = required_image_capabilities(&words);
        assert!(need.write_without_format, "write to an Unknown-format image");
        assert!(!need.extended_formats, "Unknown is in the core set");

        // And the splice puts it in, once.
        let mut patched = words.clone();
        let added = ensure_image_capabilities(&mut patched, &need);
        assert!(added.write_without_format);
        assert_eq!(patched.len(), words.len() + 2);
        assert!(!ensure_image_capabilities(&mut patched, &need).any());
        assert_eq!(patched.len(), words.len() + 2);
        // The splice lands in the capability section, before OpMemoryModel — a
        // capability declared after it is as invalid as a missing one.
        assert_eq!(patched[7] & 0xffff, OP_CAPABILITY as u32);
        assert_eq!(patched[8], 56, "SPIR-V Capability StorageImageWriteWithoutFormat");
        assert_eq!(patched[9] & 0xffff, 14, "OpMemoryModel follows");
    }

    /// A write to a *declared*-format image asks for nothing.
    #[test]
    fn a_write_to_a_declared_format_image_needs_no_capability() {
        let mut body = Vec::new();
        body.extend_from_slice(&op_type_image(10, IMAGE_SAMPLED_STORAGE, 4)); // Rgba8
        body.extend_from_slice(&[(4u32 << 16) | OP_TYPE_POINTER as u32, 11, 0, 10]);
        body.extend_from_slice(&[(4u32 << 16) | OP_VARIABLE as u32, 11, 12, 0]);
        body.extend_from_slice(&[(4u32 << 16) | OP_LOAD as u32, 10, 13, 12]);
        body.extend_from_slice(&[(4u32 << 16) | OP_IMAGE_WRITE as u32, 13, 20, 21]);
        let words = module_with(&body);
        assert!(!required_image_capabilities(&words).any());
    }

    #[test]
    fn image_format_unknown_round_trips_raw_zero() {
        // SPIR-V ImageFormat 0 is `Unknown`; it must survive a raw round trip so
        // `specialize_image_formats` can request and verify it.
        assert_eq!(ImageFormat::from_raw(0), ImageFormat::Unknown);
        assert_eq!(ImageFormat::Unknown.raw(), 0);
    }

    use metal2vulkan::meta::{FunctionConstant, TextureComponent, TextureShape};
    use metal2vulkan::reflect::{
        DescriptorLocation, ResourceBinding, ResourceKind, ShaderReflection, ShaderStage,
        REFLECTION_VERSION,
    };

    fn empty_reflection(stage: ShaderStage) -> ShaderReflection {
        ShaderReflection {
            reflection_version: REFLECTION_VERSION,
            stage,
            entry_point: None,
            bindings: vec![],
            vertex_attributes: vec![],
            varyings: vec![],
            render_targets: vec![],
            depth_members: vec![],
            stencil_members: vec![],
            local_size: None,
            vertex_builtins: None,
            imageblock_layouts: vec![],
            datalayout: None,
            function_constants: vec![],
        }
    }

    fn texture_binding(binding: u32, shape: TextureShape) -> ResourceBinding {
        ResourceBinding {
            kind: ResourceKind::Texture,
            metal_index: binding - TEXTURE_BINDING_BASE,
            descriptor: Some(DescriptorLocation { set: 0, binding }),
            param_index: None,
            address_space: None,
            declared_size: None,
            type_layout: None,
            type_name: None,
            texture_shape: Some(shape),
            embedded_source: None,
            access: None,
            static_sampler: None,
        }
    }

    /// A reflected static sampler. `binding` is the translator's number, because
    /// reflection is the translator's own output and `census_reflection_wellformed`
    /// validates it against the translator's bands.
    fn static_sampler_binding(binding: u32) -> ResourceBinding {
        use metal2vulkan::reflect::{
            SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction, SamplerCoordinates,
            SamplerFilter, SamplerMipFilter, SamplerReduction, StaticSamplerState,
        };

        ResourceBinding {
            kind: ResourceKind::StaticSampler,
            metal_index: binding - M2V_SAMPLER_BINDING_BASE,
            descriptor: Some(DescriptorLocation {
                set: RESOURCE_DESCRIPTOR_SET,
                binding,
            }),
            param_index: None,
            address_space: None,
            declared_size: None,
            type_layout: None,
            type_name: None,
            texture_shape: None,
            embedded_source: None,
            access: None,
            static_sampler: Some(StaticSamplerState {
                min_filter: SamplerFilter::Linear,
                mag_filter: SamplerFilter::Linear,
                mip_filter: SamplerMipFilter::None,
                address_mode_s: SamplerAddressMode::ClampToEdge,
                address_mode_t: SamplerAddressMode::ClampToEdge,
                address_mode_r: SamplerAddressMode::ClampToEdge,
                coordinates: SamplerCoordinates::Normalized,
                compare_function: SamplerCompareFunction::Never,
                max_anisotropy: 1,
                lod_min_clamp: 0.0,
                lod_max_clamp: 65504.0,
                border_color: SamplerBorderColor::TransparentBlack,
                reduction: SamplerReduction::WeightedAverage,
                lod_bias: 0.0,
                raw_words: [0x807b_ff00_0008_0a49, 0],
            }),
        }
    }

    fn shape(dimension: TextureDimension, arrayed: bool, writable: bool) -> TextureShape {
        TextureShape {
            dimension,
            arrayed,
            multisampled: false,
            component: TextureComponent::Float,
            writable,
            array_ref: false,
            storage_format: None,
        }
    }

    #[test]
    fn reflection_derived_kind_and_access_cover_every_shape() {
        // Dimensionality mapping matches the SPIR-V-walk `SampledImageKind`.
        let cases = [
            (TextureDimension::D1, false, Some(SampledImageKind::D1)),
            (TextureDimension::D1, true, Some(SampledImageKind::D1Array)),
            (TextureDimension::D2, false, Some(SampledImageKind::D2)),
            (TextureDimension::D2, true, Some(SampledImageKind::D2Array)),
            (TextureDimension::D3, false, Some(SampledImageKind::D3)),
            (TextureDimension::Cube, false, Some(SampledImageKind::Cube)),
            (
                TextureDimension::Cube,
                true,
                Some(SampledImageKind::CubeArray),
            ),
            // SPIR-V's SampledImageKind cannot express these — the walk returns
            // None for them too, so reflection must agree.
            (TextureDimension::D3, true, None),
            (TextureDimension::Buffer, false, None),
        ];
        for (dim, arrayed, want) in cases {
            let mut r = empty_reflection(ShaderStage::Fragment);
            let bind = TEXTURE_BINDING_BASE + 3;
            r.bindings
                .push(texture_binding(bind, shape(dim, arrayed, false)));
            assert_eq!(
                reflected_sampled_kind(&r, bind),
                want.map_or(
                    ReflectedSampledKind::Unsupported,
                    ReflectedSampledKind::Kind
                ),
                "dim={dim:?} arrayed={arrayed}"
            );
        }

        // Access mapping: writable => storage image, else sampled.
        let mut r = empty_reflection(ShaderStage::Kernel);
        let sampled = TEXTURE_BINDING_BASE;
        let storage = TEXTURE_BINDING_BASE + 1;
        r.bindings.push(texture_binding(
            sampled,
            shape(TextureDimension::D2, false, false),
        ));
        r.bindings.push(texture_binding(
            storage,
            shape(TextureDimension::D2, false, true),
        ));
        assert_eq!(
            reflected_compute_texture(&r, sampled),
            ReflectedComputeTexture::Plain2d(ImageAccess::Sampled)
        );
        assert_eq!(
            reflected_compute_texture(&r, storage),
            ReflectedComputeTexture::Plain2d(ImageAccess::Storage)
        );
        // A binding reflection does not carry => Absent (the walk's miss).
        assert_eq!(
            reflected_compute_texture(&r, TEXTURE_BINDING_BASE + 9),
            ReflectedComputeTexture::Absent
        );
        // Absent, not Unsupported: the binding is not in the reflection at all,
        // and only `reflected_sampled_kind` can tell those apart.
        assert_eq!(
            reflected_sampled_kind(&r, TEXTURE_BINDING_BASE + 9),
            ReflectedSampledKind::Absent
        );
    }

    /// The compute rail stages one flat `width × height` rectangle per texture
    /// binding, so every declared shape with a slice, depth, or sample axis must
    /// come back named rather than collapsing into the plain-2D arm — binding a
    /// `TYPE_2D` view to an image the shader declared otherwise is a
    /// descriptor-type mismatch, not a degraded render.
    #[test]
    fn every_unstageable_compute_texture_shape_names_its_axis() {
        let bind = TEXTURE_BINDING_BASE + 4;
        let unstageable = [
            (TextureDimension::D1, false, false, "dim_1d"),
            (TextureDimension::D1, true, false, "dim_1d"),
            (TextureDimension::D3, false, false, "dim_3d"),
            (TextureDimension::Cube, false, false, "dim_cube"),
            (TextureDimension::Cube, true, false, "dim_cube"),
            (TextureDimension::Buffer, false, false, "dim_buffer"),
            (TextureDimension::D2, true, false, "arrayed"),
            (TextureDimension::D2, false, true, "multisampled"),
        ];
        for (dimension, arrayed, multisampled, axis) in unstageable {
            for writable in [false, true] {
                let mut r = empty_reflection(ShaderStage::Kernel);
                let mut s = shape(dimension, arrayed, writable);
                s.multisampled = multisampled;
                r.bindings.push(texture_binding(bind, s));
                assert_eq!(
                    reflected_compute_texture(&r, bind),
                    ReflectedComputeTexture::UnstageableShape { axis },
                    "dim={dimension:?} arrayed={arrayed} ms={multisampled} writable={writable}"
                );
            }
        }

        // The one stageable shape, both access classes, is not swept up by it.
        for (writable, want) in [(false, ImageAccess::Sampled), (true, ImageAccess::Storage)] {
            let mut r = empty_reflection(ShaderStage::Kernel);
            r.bindings.push(texture_binding(
                bind,
                shape(TextureDimension::D2, false, writable),
            ));
            assert_eq!(
                reflected_compute_texture(&r, bind),
                ReflectedComputeTexture::Plain2d(want)
            );
        }
    }

    #[test]
    fn wellformed_guard_passes_consistent_reflection_and_catches_desync() {
        use metal2vulkan::reflect::ResourceAccess;

        // A consistent sampled 2D texture: kind Texture, !writable, access Sampled.
        let bind = TEXTURE_BINDING_BASE + 2;
        let mut r = empty_reflection(ShaderStage::Fragment);
        let mut b = texture_binding(bind, shape(TextureDimension::D2, false, false));
        b.access = Some(ResourceAccess::Sampled);
        r.bindings.push(b);
        assert_eq!(census_reflection_wellformed(&r, 0), 0);

        // Static samplers must carry decoded state in set 0 inside [64,96).
        let mut static_reflection = empty_reflection(ShaderStage::Fragment);
        static_reflection
            .bindings
            .push(static_sampler_binding(M2V_SAMPLER_BINDING_BASE + 1));
        assert_eq!(census_reflection_wellformed(&static_reflection, 0), 0);
        let mut missing_state = static_reflection.clone();
        missing_state.bindings[0].static_sampler = None;
        assert_eq!(census_reflection_wellformed(&missing_state, 0), 1);
        let mut out_of_band = static_reflection.clone();
        out_of_band.bindings[0].descriptor.as_mut().unwrap().binding = COLOR_INPUT_BINDING_BASE;
        assert_eq!(census_reflection_wellformed(&out_of_band, 0), 1);

        // A consistent storage image: kind StorageImage, writable, access Storage.
        let mut rs = empty_reflection(ShaderStage::Kernel);
        let mut sb = texture_binding(
            TEXTURE_BINDING_BASE + 1,
            shape(TextureDimension::D2, false, true),
        );
        sb.kind = ResourceKind::StorageImage;
        sb.access = Some(ResourceAccess::Storage);
        rs.bindings.push(sb);
        assert_eq!(census_reflection_wellformed(&rs, 0), 0);

        // Desync writable=true while kind stays Texture: one violation.
        let mut rbad = empty_reflection(ShaderStage::Fragment);
        rbad.bindings.push(texture_binding(
            bind,
            shape(TextureDimension::D2, false, true),
        ));
        assert_eq!(census_reflection_wellformed(&rbad, 0), 1);

        // Desync access=Storage while kind stays Texture: one violation.
        let mut racc = empty_reflection(ShaderStage::Fragment);
        let mut ba = texture_binding(bind, shape(TextureDimension::D2, false, false));
        ba.access = Some(ResourceAccess::Storage);
        racc.bindings.push(ba);
        assert_eq!(census_reflection_wellformed(&racc, 0), 1);

        // A stale reflection ABI version is a violation on its own.
        let mut rver = empty_reflection(ShaderStage::Fragment);
        rver.reflection_version = REFLECTION_VERSION.wrapping_add(1);
        assert_eq!(census_reflection_wellformed(&rver, 0), 1);
    }

    #[test]
    fn folded_function_constants_reported_only_when_present() {
        // FC-free shader: silent (returns 0), no analysis line.
        let none = empty_reflection(ShaderStage::Fragment);
        assert_eq!(log_folded_function_constants(&none), 0);

        // Shader declaring runtime function constants: consumed straight from the
        // reflection (single source of truth), reported once. The count is the
        // inventory length — no SPIR-V re-walk.
        let mut r = empty_reflection(ShaderStage::Kernel);
        r.entry_point = Some("gaussian_blur".to_string());
        r.function_constants = vec![
            FunctionConstant {
                index: 0,
                name: "enable_tap".to_string(),
                type_name: "i1".to_string(),
            },
            FunctionConstant {
                index: 3,
                name: "channel_count".to_string(),
                type_name: "i32".to_string(),
            },
        ];
        assert_eq!(log_folded_function_constants(&r), 2);
    }

    #[test]
    fn offset_buffer_bindings_only_in_band() {
        // Minimal fake module: header + one OpDecorate Binding 3 (4 words).
        let mut words = vec![0u32; 5];
        words[0] = 0x0723_0203; // magic-ish
                                // OpDecorate: opcode 71, wordcount 4 → word0 = (4<<16)|71
        words.push((4u32 << 16) | 71);
        words.push(1); // target id
        words.push(DECORATION_BINDING);
        words.push(3); // binding
                       // Binding 40 (texture band) must not move
        words.push((4u32 << 16) | 71);
        words.push(2);
        words.push(DECORATION_BINDING);
        words.push(40);
        let n = offset_fragment_buffer_bindings(&mut words);
        assert_eq!(n, 1);
        assert_eq!(words[8], 3 + FRAG_BUFFER_BINDING_OFFSET);
        assert_eq!(words[12], 40);
    }

    #[test]
    #[allow(
        clippy::assertions_on_constants,
        reason = "the test pins the binding-band contract constants"
    )]
    fn relocations_preserve_color_input_band_and_stay_collision_free() {
        // The ColorInput band (framebuffer-fetch SubpassData images, plus
        // today's synthesized constexpr samplers) must survive BOTH fragment
        // relocations unchanged, and the relocated buffer band must not land on
        // it — the engine binds the input attachment at its un-relocated number.
        //
        // These are DEVICE bindings, the numbering `widen_sampled_bands` leaves
        // behind, because that is what the fragment relocations run on. The
        // decorations carry no types, so this also exercises the band-predicate
        // fallback that a variable the type walk cannot name falls back to.
        let decorate = |id: u32, binding: u32| {
            vec![
                (4u32 << 16) | OP_DECORATE as u32,
                id,
                DECORATION_BINDING,
                binding,
            ]
        };
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 7, 0];
        words.extend(decorate(1, 1)); // fragment buffer → relocates
        words.extend(decorate(2, COLOR_INPUT_BINDING_BASE + 1)); // ColorInput → stays
        words.extend(decorate(3, TEXTURE_BINDING_BASE + 3)); // texture → sampled reloc
        words.extend(decorate(4, SAMPLER_BINDING_BASE)); // sampler → sampled reloc

        assert_eq!(offset_fragment_sampled_resource_bindings(&mut words), 2);
        assert_eq!(offset_fragment_buffer_bindings(&mut words), 1);

        let bindings = [words[8], words[12], words[16], words[20]];
        assert_eq!(bindings[0], 1 + FRAG_BUFFER_BINDING_OFFSET);
        assert_eq!(bindings[1], COLOR_INPUT_BINDING_BASE + 1);
        assert_eq!(
            bindings[2],
            TEXTURE_BINDING_BASE + 3 + FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
        );
        assert_eq!(
            bindings[3],
            SAMPLER_BINDING_BASE + FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
        );
        // All four distinct — no merged-set duplicate bindings.
        let mut sorted = bindings;
        sorted.sort_unstable();
        assert!(sorted.windows(2).all(|w| w[0] != w[1]));
        // Band-map invariants the engine binding math relies on.
        assert!(FRAG_BUFFER_BINDING_OFFSET >= COLOR_INPUT_BINDING_BASE + 8);
        assert!(31 + FRAG_BUFFER_BINDING_OFFSET < 32 + FRAG_SAMPLED_RESOURCE_BINDING_OFFSET);
        assert!(
            (SAMPLED_RESOURCE_BINDING_LIMIT - 1) + FRAG_SAMPLED_RESOURCE_BINDING_OFFSET < u32::MAX
        );
        // The engine's INPUT_ATTACHMENT descriptor and the m2v ColorInput band
        // are the same number by contract; the two constants live on opposite
        // sides of the runtime/engine layering, so pin their equality here.
        #[cfg(feature = "backend-vulkan")]
        assert_eq!(
            COLOR_INPUT_BINDING_BASE,
            crate::backend::vulkan::engine::COLOR_INPUT_BINDING
        );
    }

    /// Build a set-0 descriptor variable with a real type chain:
    /// `OpVariable` → `OpTypePointer(UniformConstant)` → the given type id, plus
    /// its `Binding` decoration. Ids are derived from `var` so callers only pick
    /// one number per variable.
    fn typed_descriptor(var: u32, pointee: u32, binding: u32) -> Vec<u32> {
        let pointer = var + 100;
        let mut w = vec![
            (4u32 << 16) | OP_TYPE_POINTER as u32,
            pointer,
            STORAGE_CLASS_UNIFORM_CONSTANT,
            pointee,
        ];
        w.extend([
            (4u32 << 16) | OP_VARIABLE as u32,
            pointer,
            var,
            STORAGE_CLASS_UNIFORM_CONSTANT,
        ]);
        w.extend([
            (4u32 << 16) | OP_DECORATE as u32,
            var,
            DECORATION_BINDING,
            binding,
        ]);
        w
    }

    /// `OpTypeImage` with the given result id and `Dim`. The remaining operands
    /// are a plain sampled 2D image; only `Dim` selects the class.
    fn image_type(id: u32, dim: u32) -> Vec<u32> {
        vec![
            (9u32 << 16) | OP_TYPE_IMAGE as u32,
            id,
            1, // sampled type
            dim,
            0, // depth
            0, // arrayed
            0, // multisampled
            1, // sampled
            0, // format
        ]
    }

    /// A module header whose id bound clears every id these tests use.
    fn module_header() -> Vec<u32> {
        vec![0x0723_0203, 0x0001_0000, 0, 512, 0]
    }

    /// The collision that used to bound the texture table, and the widen that
    /// resolves it.
    ///
    /// metal2vulkan puts texture `N` at `32+N` and sampler `N` at `64+N`, so
    /// texture 40 and sampler 8 are both binding 72 — one number for two
    /// descriptors, which is a module the narrow bands cannot express at all.
    /// The SPIR-V type tells them apart, and [`widen_sampled_bands`] acts on the
    /// type: the two arrive as one number and leave as two. That is what makes
    /// texture indices 32..127 reachable rather than merely countable.
    #[test]
    fn a_texture_and_a_sampler_sharing_one_binding_are_separated_by_the_widen() {
        const IMAGE: u32 = 10;
        const SAMPLER: u32 = 11;
        const COLLIDING: u32 = M2V_TEXTURE_BINDING_BASE + 40;
        const _: () = assert!(COLLIDING == M2V_SAMPLER_BINDING_BASE + 8);

        let mut words = module_header();
        words.extend(image_type(IMAGE, 1));
        words.extend([(2u32 << 16) | OP_TYPE_SAMPLER as u32, SAMPLER]);
        words.extend(typed_descriptor(30, IMAGE, COLLIDING));
        words.extend(typed_descriptor(31, SAMPLER, COLLIDING));

        let classes = variable_classes(&words);
        assert_eq!(classes[30], Some(BindingClass::Texture));
        assert_eq!(classes[31], Some(BindingClass::Sampler));

        assert_eq!(widen_sampled_bands(&mut words), 1, "only the sampler moves");
        let binding_of = |var: u32| {
            let mut i = HEADER_WORDS;
            let mut found = None;
            while i < words.len() {
                let wc = (words[i] >> 16) as usize;
                if wc == 0 || i + wc > words.len() {
                    break;
                }
                if (words[i] & 0xffff) as u16 == OP_DECORATE
                    && words[i + 2] == DECORATION_BINDING
                    && words[i + 1] == var
                {
                    found = Some(words[i + 3]);
                }
                i += wc;
            }
            found.expect("every descriptor here carries a Binding decoration")
        };
        // The texture keeps the translator's number, which is already correct in
        // a 128-wide band; the sampler moves out from under it.
        assert_eq!(binding_of(30), TEXTURE_BINDING_BASE + 40);
        assert_eq!(binding_of(31), SAMPLER_BINDING_BASE + 8);
        assert_ne!(binding_of(30), binding_of(31));
    }


    /// A framebuffer-fetch input is an `OpTypeImage` too, and the exclusion that
    /// keeps it un-relocated must not be its binding number: `Dim SubpassData`
    /// is what separates it from a texture.
    #[test]
    fn subpass_data_is_color_input_and_a_plain_image_is_a_texture() {
        const SUBPASS: u32 = 10;
        const SAMPLED: u32 = 11;
        let mut words = module_header();
        words.extend(image_type(SUBPASS, DIM_SUBPASS_DATA));
        words.extend(image_type(SAMPLED, 1));
        words.extend(typed_descriptor(30, SUBPASS, COLOR_INPUT_BINDING_BASE + 1));
        words.extend(typed_descriptor(31, SAMPLED, TEXTURE_BINDING_BASE + 3));

        let classes = variable_classes(&words);
        assert_eq!(classes[30], Some(BindingClass::ColorInput));
        assert_eq!(classes[31], Some(BindingClass::Texture));
    }

    /// The type rule replaces the band rule, so on the layout the band rule was
    /// written for the two must produce identical bindings. A module with one of
    /// each class, every index below the collision, relocated by both fragment
    /// passes.
    #[test]
    fn the_type_rule_relocates_a_typed_module_exactly_as_the_band_rule_did() {
        const IMAGE: u32 = 10;
        const SAMPLER: u32 = 11;
        const SUBPASS: u32 = 12;
        // A buffer's pointee resolves to no image or sampler type, which is what
        // makes it a buffer; id 13 is never declared as a type.
        const BUFFER_STRUCT: u32 = 13;

        let mut words = module_header();
        words.extend(image_type(IMAGE, 1));
        words.extend([(2u32 << 16) | OP_TYPE_SAMPLER as u32, SAMPLER]);
        words.extend(image_type(SUBPASS, DIM_SUBPASS_DATA));
        words.extend(typed_descriptor(30, BUFFER_STRUCT, 3));
        words.extend(typed_descriptor(31, IMAGE, TEXTURE_BINDING_BASE + 3));
        words.extend(typed_descriptor(32, SAMPLER, SAMPLER_BINDING_BASE + 2));
        words.extend(typed_descriptor(33, SUBPASS, COLOR_INPUT_BINDING_BASE + 1));

        assert_eq!(offset_fragment_sampled_resource_bindings(&mut words), 2);
        assert_eq!(offset_fragment_buffer_bindings(&mut words), 1);

        // Each `typed_descriptor` ends with its 4-word decoration, so the
        // binding is the last word of each block.
        let binding_of = |var: u32| {
            let mut i = HEADER_WORDS;
            while i < words.len() {
                let word_count = (words[i] >> 16) as usize;
                if (words[i] & 0xffff) as u16 == OP_DECORATE
                    && words[i + 2] == DECORATION_BINDING
                    && words[i + 1] == var
                {
                    return words[i + 3];
                }
                i += word_count;
            }
            unreachable!("every variable in this module carries a Binding");
        };
        assert_eq!(binding_of(30), 3 + FRAG_BUFFER_BINDING_OFFSET);
        assert_eq!(
            binding_of(31),
            TEXTURE_BINDING_BASE + 3 + FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
        );
        assert_eq!(
            binding_of(32),
            SAMPLER_BINDING_BASE + 2 + FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
        );
        assert_eq!(
            binding_of(33),
            COLOR_INPUT_BINDING_BASE + 1,
            "the ColorInput band never moves"
        );
    }

    #[test]
    fn reflects_storage_image_format_without_names() {
        // A storage image at binding 34 with an explicit Rgba8Uint format operand.
        // `image_format` stays a structural SPIR-V walk (reflection carries no
        // explicit storage-format for the format-specialization path).
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 7, 0];
        words.extend([
            (9u32 << 16) | OP_TYPE_IMAGE as u32,
            4,
            99,
            1,
            0,
            0,
            0,
            2,
            32,
        ]);
        words.extend([(4u32 << 16) | OP_TYPE_POINTER as u32, 5, 0, 4]);
        words.extend([(4u32 << 16) | OP_VARIABLE as u32, 5, 6, 0]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 6, DECORATION_BINDING, 34]);
        assert_eq!(image_format(&words, 34), Some(ImageFormat::Rgba8Uint));
        assert_eq!(image_format(&words, 33), None);
    }

    #[test]
    fn reflects_storage_image_content_access_without_names() {
        // %1=image type, %2=pointer, %3=variable(binding 34), %4=loaded
        // image, %5=read result. The first module reads and writes; removing
        // OpImageRead proves write-only access without decorations or names.
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 6, 0];
        words.extend([
            (9u32 << 16) | OP_TYPE_IMAGE as u32,
            1,
            99,
            1,
            0,
            0,
            0,
            2,
            32,
        ]);
        words.extend([
            (4u32 << 16) | OP_TYPE_POINTER as u32,
            2,
            STORAGE_CLASS_UNIFORM_CONSTANT,
            1,
        ]);
        words.extend([
            (4u32 << 16) | OP_VARIABLE as u32,
            2,
            3,
            STORAGE_CLASS_UNIFORM_CONSTANT,
        ]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 3, DECORATION_BINDING, 34]);
        words.extend([(4u32 << 16) | OP_LOAD as u32, 1, 4, 3]);
        let write_at = words.len();
        words.extend([(4u32 << 16) | OP_IMAGE_WRITE as u32, 4, 90, 91]);
        assert_eq!(
            storage_image_access(&words, 34),
            Some(StorageImageAccess::WriteOnly)
        );
        words.splice(
            write_at..write_at,
            [(5u32 << 16) | OP_IMAGE_READ as u32, 92, 5, 4, 90],
        );
        assert_eq!(
            storage_image_access(&words, 34),
            Some(StorageImageAccess::ReadWrite)
        );

        words.extend([
            (4u32 << 16) | OP_VARIABLE as u32,
            2,
            5,
            STORAGE_CLASS_UNIFORM_CONSTANT,
        ]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 5, DECORATION_BINDING, 34]);
        assert_eq!(
            storage_image_access(&words, 34),
            Some(StorageImageAccess::AmbiguousBinding)
        );
    }

    /// The two reflectors share `propagate_derived` but seed it differently, and
    /// the difference is not cosmetic.
    ///
    /// `buffer_access` tracks pointers and seeds the descriptor variable itself;
    /// its escape scan enumerates the opcodes it cares about, so naming the
    /// variable elsewhere is harmless. `storage_image_access` tracks image
    /// *values* and must not seed the variable, because its escape scan ends in a
    /// catch-all that treats any instruction mentioning a tracked id as an
    /// escape. `OpDecorate` names the variable in every real module, so a seeded
    /// root would make every storage image reflect `Unknown` — silently, and in
    /// the direction that looks like caution.
    ///
    /// This module is the ordinary read/write case plus the `OpEntryPoint`
    /// interface list, which is the second place a variable id appears.
    #[test]
    fn storage_image_root_is_not_seeded_so_naming_the_variable_is_not_an_escape() {
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 6, 0];
        words.extend([
            (9u32 << 16) | OP_TYPE_IMAGE as u32,
            1,
            99,
            1,
            0,
            0,
            0,
            2,
            32,
        ]);
        words.extend([
            (4u32 << 16) | OP_TYPE_POINTER as u32,
            2,
            STORAGE_CLASS_UNIFORM_CONSTANT,
            1,
        ]);
        words.extend([
            (4u32 << 16) | OP_VARIABLE as u32,
            2,
            3,
            STORAGE_CLASS_UNIFORM_CONSTANT,
        ]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 3, DECORATION_BINDING, 34]);
        // OpEntryPoint's interface list names every module-scope variable.
        words.extend([(5u32 << 16) | 15u32, 5, 100, 0, 3]);
        words.extend([(4u32 << 16) | OP_LOAD as u32, 1, 4, 3]);
        words.extend([(5u32 << 16) | OP_IMAGE_READ as u32, 92, 5, 4, 90]);
        words.extend([(4u32 << 16) | OP_IMAGE_WRITE as u32, 4, 90, 91]);
        assert_eq!(
            storage_image_access(&words, 34),
            Some(StorageImageAccess::ReadWrite),
            "the variable being decorated and listed as an interface is not an escape"
        );
    }

    /// Every `OpLoad` in `words` whose pointer is a global `OpVariable` must
    /// declare the pointee of that variable's pointer type as its result type.
    ///
    /// This is a SPIR-V validity rule — "Result Type must be the same as the
    /// type pointed to by Pointer" — and it is the one that a format
    /// specialization can break without touching a single instruction inside
    /// the function: repointing a variable at a cloned image type leaves every
    /// load of it declaring the type the variable no longer has. Asserted
    /// structurally rather than by shelling out to `spirv-val`, so it runs on
    /// every arm and needs nothing installed.
    fn loads_agree_with_their_pointees(words: &[u32]) -> Result<(), String> {
        let bound = words[3] as usize;
        let mut pointer_pointee = vec![None; bound];
        let mut variable_type = vec![None; bound];
        let mut i = HEADER_WORDS;
        while i < words.len() {
            let count = (words[i] >> 16) as usize;
            let opcode = (words[i] & 0xffff) as u16;
            assert!(count > 0 && i + count <= words.len(), "malformed module");
            match opcode {
                OP_TYPE_POINTER if count >= 4 => {
                    pointer_pointee[words[i + 1] as usize] = Some(words[i + 3] as usize)
                }
                OP_VARIABLE if count >= 4 => {
                    variable_type[words[i + 2] as usize] = Some(words[i + 1] as usize)
                }
                _ => {}
            }
            i += count;
        }
        let mut i = HEADER_WORDS;
        while i < words.len() {
            let count = (words[i] >> 16) as usize;
            let opcode = (words[i] & 0xffff) as u16;
            if opcode == OP_LOAD && count >= 4 {
                let result_type = words[i + 1] as usize;
                let pointer = words[i + 3] as usize;
                let pointee = variable_type
                    .get(pointer)
                    .copied()
                    .flatten()
                    .and_then(|p| pointer_pointee.get(p).copied().flatten());
                if let Some(pointee) = pointee {
                    if pointee != result_type {
                        return Err(format!(
                            "OpLoad of %{pointer} declares %{result_type} but the variable now points at %{pointee}"
                        ));
                    }
                }
            }
            i += count;
        }
        Ok(())
    }

    /// Two storage images sharing one `OpTypeImage`, only one of them
    /// specialized, each loaded and written in the function body.
    ///
    /// The clone path is the only one that changes a variable's *type*, and it
    /// is the shape a real kernel takes: `metal2vulkan` emits one `OpTypeImage`
    /// per (sampled type, dimension, format) tuple, so two
    /// `texture2d<float, access::write>` parameters share it, and the device
    /// specializes only the binding the guest bound a surface to.
    ///
    /// Left unrepaired, the resulting module is invalid and the NVIDIA driver's
    /// SPIR-V compiler segmentation-faults inside `vkCreateComputePipelines` —
    /// taking QEMU with it. That is what the macos-14 rail did on its first
    /// compute dispatch, and a guest can author any kernel, so a module this
    /// device assembled must never be one the driver cannot survive.
    #[test]
    fn a_cloned_image_type_carries_its_loads_with_it() {
        // %99 float, %1 OpTypeImage(float, format R32f), %2 pointer to %1,
        // %3/%4 variables at bindings 34/35, then a function that loads both.
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 10, 0];
        words.extend([(9u32 << 16) | OP_TYPE_IMAGE as u32, 1, 99, 1, 0, 0, 0, 2, 3]);
        words.extend([(4u32 << 16) | OP_TYPE_POINTER as u32, 2, 0, 1]);
        words.extend([(4u32 << 16) | OP_VARIABLE as u32, 2, 3, 0]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 3, DECORATION_BINDING, 34]);
        words.extend([(4u32 << 16) | OP_VARIABLE as u32, 2, 4, 0]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 4, DECORATION_BINDING, 35]);
        words.extend([(5u32 << 16) | OP_FUNCTION as u32, 98, 5, 0, 97]);
        words.extend([(4u32 << 16) | OP_LOAD as u32, 1, 6, 3]);
        words.extend([(4u32 << 16) | OP_LOAD as u32, 1, 7, 4]);

        assert!(loads_agree_with_their_pointees(&words).is_ok(), "premise");
        assert_eq!(
            specialize_image_formats(&mut words, &[(34, ImageFormat::Rgba16Float)]),
            Ok(1)
        );
        assert_eq!(image_format(&words, 34), Some(ImageFormat::Rgba16Float));
        assert_eq!(image_format(&words, 35), Some(ImageFormat::R32Float));
        if let Err(why) = loads_agree_with_their_pointees(&words) {
            panic!("specialization left the module invalid: {why}");
        }
    }

    /// The smallest module a Vulkan validator accepts: an empty `GLCompute`
    /// entry point with a declared local size.
    fn minimal_compute_module() -> Vec<u32> {
        const OP_CAPABILITY: u32 = 17;
        const OP_MEMORY_MODEL: u32 = 14;
        const OP_ENTRY_POINT: u32 = 15;
        const OP_EXECUTION_MODE: u32 = 16;
        const OP_TYPE_VOID: u32 = 19;
        const OP_TYPE_FUNCTION: u32 = 33;
        const OP_FUNCTION: u32 = 54;
        const OP_FUNCTION_END: u32 = 56;
        const OP_LABEL: u32 = 248;
        const OP_RETURN: u32 = 253;
        let mut w = vec![0x0723_0203, 0x0001_0000, 0, 5, 0];
        w.extend([(2 << 16) | OP_CAPABILITY, 1]); // Shader
        w.extend([(3 << 16) | OP_MEMORY_MODEL, 0, 1]); // Logical GLSL450
        w.extend([(5 << 16) | OP_ENTRY_POINT, 5, 3, 0x6E69_616D, 0]); // GLCompute %3 "main"
        w.extend([(6 << 16) | OP_EXECUTION_MODE, 3, 17, 1, 1, 1]); // LocalSize 1 1 1
        w.extend([(2 << 16) | OP_TYPE_VOID, 1]);
        w.extend([(3 << 16) | OP_TYPE_FUNCTION, 2, 1]);
        w.extend([(5 << 16) | OP_FUNCTION, 1, 3, 0, 2]);
        w.extend([(2 << 16) | OP_LABEL, 4]);
        w.extend([1 << 16 | OP_RETURN]);
        w.extend([1 << 16 | OP_FUNCTION_END]);
        w
    }

    /// The gate that stands between a guest's shader and undefined behaviour
    /// inside a driver.
    ///
    /// A valid module passes and an invalid one is named, which is the whole
    /// contract — the device declines the dispatch rather than handing the
    /// driver something it is entitled to crash on, and one driver does.
    ///
    /// Skips where SPIRV-Tools is not installed, because there the function is
    /// specified to accept: an absent instrument is not a verdict.
    #[test]
    fn a_module_a_validator_rejects_never_reaches_a_driver() {
        let good = minimal_compute_module();
        if validate(&good) != SpirvValidation::Accepted {
            eprintln!("skip: no usable spirv-val, or it rejects the minimal module");
            return;
        }
        // Point the function's return type at the function id instead of at
        // %void, which is a type error no driver has to survive.
        let mut bad = good.clone();
        let at = bad
            .iter()
            .position(|w| *w == (5 << 16) | 54)
            .expect("OpFunction");
        bad[at + 1] = 3;
        match validate(&bad) {
            SpirvValidation::Rejected(why) => assert!(!why.is_empty(), "the refusal must say why"),
            SpirvValidation::Accepted => panic!("a type-broken module was accepted"),
        }
    }

    /// Concurrent validations do not answer for each other.
    ///
    /// The validator wrapper writes a fixed file name inside the directory it
    /// is handed, so a shared directory makes two callers race over one path
    /// and the loser gets a verdict about bytes it did not submit. That is not
    /// hypothetical: it rejected three good modules on a working macos-13 boot,
    /// where the concurrency comes from this device's drain thread and
    /// `metal2vulkan`'s own async translations sharing `/tmp`.
    ///
    /// Skips where SPIRV-Tools is absent, where every answer is `Accepted` and
    /// the test could not fail.
    #[test]
    fn concurrent_validations_do_not_collide() {
        let good = minimal_compute_module();
        if validate(&good) != SpirvValidation::Accepted {
            eprintln!("skip: no usable spirv-val");
            return;
        }
        let verdicts: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let words = good.clone();
                    scope.spawn(move || validate(&words))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for verdict in &verdicts {
            assert_eq!(*verdict, SpirvValidation::Accepted, "{verdicts:?}");
        }
    }

    /// The guard refuses rather than trusting the repair.
    ///
    /// `retype_loads` handles the shape the clone path produces; the check is
    /// what makes anything it did not handle a decline instead of a module
    /// handed to a driver that may not survive it. Fed a module broken the way
    /// an unrepaired clone breaks one, it names the load.
    #[test]
    fn a_load_that_disagrees_with_its_pointee_is_refused_by_name() {
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 10, 0];
        words.extend([(9u32 << 16) | OP_TYPE_IMAGE as u32, 1, 99, 1, 0, 0, 0, 2, 3]);
        words.extend([(9u32 << 16) | OP_TYPE_IMAGE as u32, 8, 99, 1, 0, 0, 0, 2, 2]);
        words.extend([(4u32 << 16) | OP_TYPE_POINTER as u32, 2, 0, 8]);
        words.extend([(4u32 << 16) | OP_VARIABLE as u32, 2, 3, 0]);
        // The variable points at %8 and the load still declares %1.
        words.extend([(4u32 << 16) | OP_LOAD as u32, 1, 6, 3]);

        assert_eq!(
            verify_load_types(&words),
            Err(ImageFormatSpecializeError::LoadTypeMismatch {
                pointer: 3,
                declared: 1
            })
        );
        assert_eq!(
            crate::observe::Decline::slug(&verify_load_types(&words).unwrap_err()),
            "spirv_format_specialize_load_type_mismatch"
        );
        // And it passes once the load names what the variable points at.
        let at = words.len() - 3;
        words[at] = 8;
        assert_eq!(verify_load_types(&words), Ok(()));
    }

    #[test]
    fn specializes_image_format_by_binding_without_names() {
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 7, 0];
        words.extend([(9u32 << 16) | OP_TYPE_IMAGE as u32, 1, 99, 1, 0, 0, 0, 2, 1]);
        words.extend([(4u32 << 16) | OP_TYPE_POINTER as u32, 2, 0, 1]);
        words.extend([(4u32 << 16) | OP_VARIABLE as u32, 2, 3, 0]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 3, DECORATION_BINDING, 34]);
        words.extend([(4u32 << 16) | OP_VARIABLE as u32, 2, 4, 0]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 4, DECORATION_BINDING, 35]);

        assert_eq!(
            specialize_image_formats(&mut words, &[(34, ImageFormat::Rgba16Float)]),
            Ok(1)
        );
        assert_eq!(image_format(&words, 34), Some(ImageFormat::Rgba16Float));
        assert_eq!(image_format(&words, 35), Some(ImageFormat::Rgba32Float));
        assert_eq!(
            specialize_image_formats(
                &mut words,
                &[
                    (34, ImageFormat::Rgba16Float),
                    (35, ImageFormat::Rgba8Unorm)
                ]
            ),
            Ok(1)
        );
        assert_eq!(image_format(&words, 34), Some(ImageFormat::Rgba16Float));
        assert_eq!(image_format(&words, 35), Some(ImageFormat::Rgba8Unorm));
    }

    #[test]
    fn specializes_rgba8ui_write_image_to_r32ui() {
        // The exact device patch for an R32Uint-bound `texture2d<uint, write>`:
        // the translator declares the storage image `Rgba8ui` (SPIR-V format
        // token 32); the device re-targets it to `R32ui` (token 33) so the view
        // is VK_FORMAT_R32_UINT. Verify the reflection reads Rgba8ui, the patch
        // rewrites the format operand, and it reads back as R32ui.
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 6, 0];
        // OpTypeImage %1 : uint 2D depth=0 arrayed=0 ms=0 sampled=2 format=32.
        words.extend([
            (9u32 << 16) | OP_TYPE_IMAGE as u32,
            1,
            99,
            1,
            0,
            0,
            0,
            2,
            32,
        ]);
        words.extend([(4u32 << 16) | OP_TYPE_POINTER as u32, 2, 0, 1]);
        words.extend([(4u32 << 16) | OP_VARIABLE as u32, 2, 3, 0]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 3, DECORATION_BINDING, 33]);

        assert_eq!(image_format(&words, 33), Some(ImageFormat::Rgba8Uint));
        assert_eq!(ImageFormat::R32ui.raw(), 33);
        assert_eq!(ImageFormat::from_raw(33), ImageFormat::R32ui);
        assert_eq!(
            specialize_image_formats(&mut words, &[(33, ImageFormat::R32ui)]),
            Ok(1)
        );
        assert_eq!(image_format(&words, 33), Some(ImageFormat::R32ui));
    }

    /// SPIR-V `R32f` is enum value 3 and is what `metal2vulkan` declares for a
    /// generic `texture2d<float, access::write>`. Leaving 3 out of the decode
    /// table turned every such storage image into `Unsupported(3)`, which the
    /// device cannot specialize — the dispatch was dropped rather than run
    /// against the bound guest surface.
    #[test]
    fn r32f_write_image_decodes_and_specializes() {
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 6, 0];
        // OpTypeImage %1 : float 2D depth=0 arrayed=0 ms=0 sampled=2 format=3.
        words.extend([(9u32 << 16) | OP_TYPE_IMAGE as u32, 1, 99, 1, 0, 0, 0, 2, 3]);
        words.extend([(4u32 << 16) | OP_TYPE_POINTER as u32, 2, 0, 1]);
        words.extend([(4u32 << 16) | OP_VARIABLE as u32, 2, 3, 0]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 3, DECORATION_BINDING, 33]);

        assert_eq!(ImageFormat::from_raw(3), ImageFormat::R32Float);
        assert_eq!(ImageFormat::R32Float.raw(), 3);
        assert_eq!(image_format(&words, 33), Some(ImageFormat::R32Float));
        assert_eq!(
            specialize_image_formats(&mut words, &[(33, ImageFormat::Rgba32Float)]),
            Ok(1)
        );
        assert_eq!(image_format(&words, 33), Some(ImageFormat::Rgba32Float));
    }

    fn storage_buffer_module(binding: u32) -> Vec<u32> {
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 12, 0];
        words.extend([
            (4u32 << 16) | OP_VARIABLE as u32,
            1,
            2,
            STORAGE_CLASS_STORAGE_BUFFER,
        ]);
        words.extend([
            (4u32 << 16) | OP_DECORATE as u32,
            2,
            DECORATION_BINDING,
            binding,
        ]);
        // %4 = access-chain %2; pointer provenance must reach the leaf.
        words.extend([(5u32 << 16) | OP_ACCESS_CHAIN as u32, 3, 4, 2, 5]);
        words
    }

    #[test]
    fn reflects_storage_buffer_read_only_without_names() {
        let words = storage_buffer_module(1);
        assert_eq!(buffer_access(&words, 1), Some(BufferAccess::ReadOnly));
        assert_eq!(buffer_access(&words, 0), None);
    }

    #[test]
    fn reflects_storage_buffer_write_through_access_chain() {
        let mut words = storage_buffer_module(1);
        words.extend([(3u32 << 16) | OP_STORE as u32, 4, 6]);
        assert_eq!(buffer_access(&words, 1), Some(BufferAccess::Writable));
    }

    /// A module the walk cannot finish reflects nothing, and does not take the
    /// process with it.
    ///
    /// Both shapes are what [`instructions`] exists to reject, and both are
    /// reached through a module that is otherwise perfectly reflectable — the
    /// binding, the variable and the storage class are all present and correct,
    /// so nothing but the malformed tail can be what turns the answer into
    /// `None`.
    ///
    /// This pins a contract rather than proving a repair: before the walk was
    /// consolidated the same two inputs also answered `None`, because
    /// `descriptor_root` rejected them on the way past. What changes is where
    /// that can be undone from. The provenance scans indexed
    /// `words[at + 1 .. at + word_count]` with no guard of their own, so
    /// removing or reordering `descriptor_root`'s left the first case aborting
    /// the device on an out-of-range slice and the second spinning forever on
    /// `i += 0`. Neither failure has a test that can run *after* it happens.
    #[test]
    fn a_module_that_does_not_walk_reflects_nothing() {
        for (what, tail) in [
            // Claims six words with two left in the module.
            ("runs past the end", vec![(6u32 << 16) | OP_STORE as u32, 4]),
            // A zero word count advances the cursor by nothing.
            ("zero word count", vec![0u32, 4]),
        ] {
            let mut words = storage_buffer_module(1);
            assert_eq!(
                buffer_access(&words, 1),
                Some(BufferAccess::ReadOnly),
                "{what}: the module is reflectable before the tail is appended"
            );
            words.extend(tail);
            assert_eq!(buffer_access(&words, 1), None, "{what}: buffer_access");
            assert_eq!(
                storage_image_access(&words, 1),
                None,
                "{what}: storage_image_access"
            );
            assert!(instructions(&words).is_none(), "{what}: instructions");
        }
    }

    #[test]
    fn storage_buffer_pointer_call_fails_access_closed() {
        let mut words = storage_buffer_module(1);
        // OpFunctionCall result-type, result-id, function-id, pointer arg.
        words.extend([(5u32 << 16) | OP_FUNCTION_CALL as u32, 7, 8, 9, 4]);
        assert_eq!(buffer_access(&words, 1), Some(BufferAccess::PointerEscape));
    }

    #[test]
    fn duplicate_storage_buffer_binding_fails_access_closed() {
        let mut words = storage_buffer_module(1);
        words.extend([
            (4u32 << 16) | OP_VARIABLE as u32,
            1,
            6,
            STORAGE_CLASS_STORAGE_BUFFER,
        ]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 6, DECORATION_BINDING, 1]);
        assert_eq!(
            buffer_access(&words, 1),
            Some(BufferAccess::AmbiguousBinding)
        );
    }

    #[test]
    fn reflects_only_declared_sampler_bindings_without_names() {
        // %1 sampler, %2 UniformConstant pointer, %3 sampler variable. A
        // decorated non-sampler %4 must not appear in the result.
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 5, 0];
        words.extend([(2u32 << 16) | OP_TYPE_SAMPLER as u32, 1]);
        words.extend([
            (4u32 << 16) | OP_TYPE_POINTER as u32,
            2,
            STORAGE_CLASS_UNIFORM_CONSTANT,
            1,
        ]);
        words.extend([
            (4u32 << 16) | OP_VARIABLE as u32,
            2,
            3,
            STORAGE_CLASS_UNIFORM_CONSTANT,
        ]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 3, DECORATION_BINDING, 66]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 4, DECORATION_BINDING, 99]);
        assert_eq!(sampler_bindings(&words), vec![66]);
    }
}
