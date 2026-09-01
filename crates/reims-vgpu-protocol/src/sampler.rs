//! `MTLSamplerDescriptor`'s ordinals, and the combinations the API itself does
//! not admit.
//!
//! # Why the ordinals are parsed once
//!
//! A sampler is six enumerations, two clamps, an anisotropy and three flags,
//! and every one of the enumerations has a closed set. Folding an unrecognised
//! address mode onto `ClampToEdge` samples the edge texel where the guest
//! wanted a repeat — a visible, plausible, unattributable difference — so each
//! parses to `None` outside its set and the refusal carries the ordinal.
//!
//! # Unnormalized coordinates are a mode, not a flag
//!
//! `MTLSamplerDescriptor.normalizedCoordinates = NO` puts the sampler in a
//! restricted mode in both APIs: one mip level, no anisotropy, no comparison,
//! and only the clamping address modes. Metal documents those restrictions and
//! Vulkan makes them validation rules, so a descriptor that breaks one is not
//! a sampler either API will build. [`SamplerShape::checked`] refuses it here,
//! with the field that broke it, rather than letting each backend discover a
//! different subset of the rules.
//!
//! # What is deliberately not here
//!
//! Whether a *host* can build the sampler. Anisotropy limits, mirror-clamp
//! support and border-colour kinds are physical-device properties and belong
//! to the executor that queried one. This layer answers only what the guest
//! API itself admits.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// `MTLSamplerMinMagFilter`.
pub const MTL_SAMPLER_MIN_MAG_FILTER_NEAREST: u32 = 0;
pub const MTL_SAMPLER_MIN_MAG_FILTER_LINEAR: u32 = 1;

/// `MTLSamplerMipFilter`.
pub const MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED: u32 = 0;
pub const MTL_SAMPLER_MIP_FILTER_NEAREST: u32 = 1;
pub const MTL_SAMPLER_MIP_FILTER_LINEAR: u32 = 2;

/// `MTLSamplerAddressMode`.
pub const MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: u32 = 0;
pub const MTL_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE: u32 = 1;
pub const MTL_SAMPLER_ADDRESS_MODE_REPEAT: u32 = 2;
pub const MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT: u32 = 3;
pub const MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO: u32 = 4;
pub const MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER_COLOR: u32 = 5;

/// `MTLSamplerBorderColor`.
pub const MTL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK: u32 = 0;
pub const MTL_SAMPLER_BORDER_COLOR_OPAQUE_BLACK: u32 = 1;
pub const MTL_SAMPLER_BORDER_COLOR_OPAQUE_WHITE: u32 = 2;

/// `MTLCompareFunction`. Shared with the depth-stencil state, which is why it
/// is spelled once.
pub const MTL_COMPARE_FUNCTION_NEVER: u32 = 0;
pub const MTL_COMPARE_FUNCTION_LESS: u32 = 1;
pub const MTL_COMPARE_FUNCTION_EQUAL: u32 = 2;
pub const MTL_COMPARE_FUNCTION_LESS_EQUAL: u32 = 3;
pub const MTL_COMPARE_FUNCTION_GREATER: u32 = 4;
pub const MTL_COMPARE_FUNCTION_NOT_EQUAL: u32 = 5;
pub const MTL_COMPARE_FUNCTION_GREATER_EQUAL: u32 = 6;
pub const MTL_COMPARE_FUNCTION_ALWAYS: u32 = 7;

/// Minification and magnification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Filter {
    Nearest,
    Linear,
}

impl Filter {
    pub const ALL: [Filter; 2] = [Self::Nearest, Self::Linear];

    #[must_use]
    pub const fn parse(ordinal: u32) -> Option<Self> {
        Some(match ordinal {
            MTL_SAMPLER_MIN_MAG_FILTER_NEAREST => Self::Nearest,
            MTL_SAMPLER_MIN_MAG_FILTER_LINEAR => Self::Linear,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Linear => "linear",
        }
    }
}

/// How levels of a mip chain are combined.
///
/// `NotMipmapped` is a third answer and not a filter: it says only the top
/// level is ever sampled, which is a clamp on the level of detail rather than
/// a choice of how to blend two levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MipFilter {
    NotMipmapped,
    Nearest,
    Linear,
}

impl MipFilter {
    pub const ALL: [MipFilter; 3] = [Self::NotMipmapped, Self::Nearest, Self::Linear];

    #[must_use]
    pub const fn parse(ordinal: u32) -> Option<Self> {
        Some(match ordinal {
            MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED => Self::NotMipmapped,
            MTL_SAMPLER_MIP_FILTER_NEAREST => Self::Nearest,
            MTL_SAMPLER_MIP_FILTER_LINEAR => Self::Linear,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::NotMipmapped => "not_mipmapped",
            Self::Nearest => "nearest",
            Self::Linear => "linear",
        }
    }
}

/// What happens outside `[0, 1]` on one axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AddressMode {
    ClampToEdge,
    MirrorClampToEdge,
    Repeat,
    MirrorRepeat,
    /// Clamp to transparent black. A border mode whose border is fixed, which
    /// is why it is distinct from [`Self::ClampToBorderColor`] rather than a
    /// spelling of it.
    ClampToZero,
    ClampToBorderColor,
}

impl AddressMode {
    pub const ALL: [AddressMode; 6] = [
        Self::ClampToEdge,
        Self::MirrorClampToEdge,
        Self::Repeat,
        Self::MirrorRepeat,
        Self::ClampToZero,
        Self::ClampToBorderColor,
    ];

    #[must_use]
    pub const fn parse(ordinal: u32) -> Option<Self> {
        Some(match ordinal {
            MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE => Self::ClampToEdge,
            MTL_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE => Self::MirrorClampToEdge,
            MTL_SAMPLER_ADDRESS_MODE_REPEAT => Self::Repeat,
            MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT => Self::MirrorRepeat,
            MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO => Self::ClampToZero,
            MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER_COLOR => Self::ClampToBorderColor,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ClampToEdge => "clamp_to_edge",
            Self::MirrorClampToEdge => "mirror_clamp_to_edge",
            Self::Repeat => "repeat",
            Self::MirrorRepeat => "mirror_repeat",
            Self::ClampToZero => "clamp_to_zero",
            Self::ClampToBorderColor => "clamp_to_border_color",
        }
    }

    /// Whether this mode reads a border rather than a texel of the image.
    #[must_use]
    pub const fn uses_border(self) -> bool {
        matches!(self, Self::ClampToZero | Self::ClampToBorderColor)
    }

    /// Whether an unnormalized-coordinate sampler may use this mode.
    ///
    /// Only the clamping modes: a repeat over unnormalized coordinates has no
    /// period to repeat over. Both APIs say so, and the restriction is the
    /// same one, so it is stated once here.
    #[must_use]
    pub const fn allows_unnormalized(self) -> bool {
        matches!(
            self,
            Self::ClampToEdge | Self::ClampToZero | Self::ClampToBorderColor
        )
    }
}

/// The border a border mode reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BorderColor {
    TransparentBlack,
    OpaqueBlack,
    OpaqueWhite,
}

impl BorderColor {
    pub const ALL: [BorderColor; 3] =
        [Self::TransparentBlack, Self::OpaqueBlack, Self::OpaqueWhite];

    #[must_use]
    pub const fn parse(ordinal: u32) -> Option<Self> {
        Some(match ordinal {
            MTL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK => Self::TransparentBlack,
            MTL_SAMPLER_BORDER_COLOR_OPAQUE_BLACK => Self::OpaqueBlack,
            MTL_SAMPLER_BORDER_COLOR_OPAQUE_WHITE => Self::OpaqueWhite,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::TransparentBlack => "transparent_black",
            Self::OpaqueBlack => "opaque_black",
            Self::OpaqueWhite => "opaque_white",
        }
    }
}

/// `MTLCompareFunction`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

impl CompareFunction {
    pub const ALL: [CompareFunction; 8] = [
        Self::Never,
        Self::Less,
        Self::Equal,
        Self::LessEqual,
        Self::Greater,
        Self::NotEqual,
        Self::GreaterEqual,
        Self::Always,
    ];

    #[must_use]
    pub const fn parse(ordinal: u32) -> Option<Self> {
        Some(match ordinal {
            MTL_COMPARE_FUNCTION_NEVER => Self::Never,
            MTL_COMPARE_FUNCTION_LESS => Self::Less,
            MTL_COMPARE_FUNCTION_EQUAL => Self::Equal,
            MTL_COMPARE_FUNCTION_LESS_EQUAL => Self::LessEqual,
            MTL_COMPARE_FUNCTION_GREATER => Self::Greater,
            MTL_COMPARE_FUNCTION_NOT_EQUAL => Self::NotEqual,
            MTL_COMPARE_FUNCTION_GREATER_EQUAL => Self::GreaterEqual,
            MTL_COMPARE_FUNCTION_ALWAYS => Self::Always,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn ordinal(self) -> u32 {
        match self {
            Self::Never => MTL_COMPARE_FUNCTION_NEVER,
            Self::Less => MTL_COMPARE_FUNCTION_LESS,
            Self::Equal => MTL_COMPARE_FUNCTION_EQUAL,
            Self::LessEqual => MTL_COMPARE_FUNCTION_LESS_EQUAL,
            Self::Greater => MTL_COMPARE_FUNCTION_GREATER,
            Self::NotEqual => MTL_COMPARE_FUNCTION_NOT_EQUAL,
            Self::GreaterEqual => MTL_COMPARE_FUNCTION_GREATER_EQUAL,
            Self::Always => MTL_COMPARE_FUNCTION_ALWAYS,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Never => "never",
            Self::Less => "less",
            Self::Equal => "equal",
            Self::LessEqual => "less_equal",
            Self::Greater => "greater",
            Self::NotEqual => "not_equal",
            Self::GreaterEqual => "greater_equal",
            Self::Always => "always",
        }
    }
}

/// A sampler declaration as the fields arrived.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SamplerShape {
    pub min_filter: u32,
    pub mag_filter: u32,
    pub mip_filter: u32,
    pub s_address: u32,
    pub t_address: u32,
    pub r_address: u32,
    /// Floored at one by the decoder. One is "no anisotropic filtering".
    pub max_anisotropy: u32,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    /// `MTLCompareFunction` ordinal. A sampler with no comparison is
    /// `Never` in Metal's own default, so this is not an `Option` on the wire
    /// — see [`SamplerState::compare`].
    pub compare_function: u32,
    /// Whether a comparison is performed at all. Metal's descriptor has no
    /// separate flag; the guest's serializer does, so it is carried rather
    /// than inferred from the function.
    pub compare_enabled: bool,
    pub border_color: u32,
    pub normalized_coordinates: bool,
}

/// Why a sampler declaration is not one the guest API admits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerRefusal {
    UnknownOrdinal {
        field: &'static str,
        ordinal: u32,
    },
    /// An unnormalized-coordinate sampler broke one of the restrictions both
    /// APIs place on it.
    UnnormalizedRestriction {
        field: &'static str,
    },
    /// The level-of-detail clamp is inverted or is not a number.
    BadLodClamp {
        min_bits: u32,
        max_bits: u32,
    },
}

impl reims_vgpu_observe::Decline for SamplerRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::UnknownOrdinal { .. } => "sampler_unknown_ordinal",
            Self::UnnormalizedRestriction { .. } => "sampler_unnormalized_restriction",
            Self::BadLodClamp { .. } => "sampler_bad_lod_clamp",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::UnknownOrdinal { field, ordinal } => vec![
                ("field", (*field).to_string()),
                ("ordinal", ordinal.to_string()),
            ],
            Self::UnnormalizedRestriction { field } => vec![("field", (*field).to_string())],
            Self::BadLodClamp { min_bits, max_bits } => vec![
                ("min_bits", min_bits.to_string()),
                ("max_bits", max_bits.to_string()),
            ],
        }
    }
}

/// A sampler declaration whose fields have been parsed and checked against
/// each other.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerState {
    min_filter: Filter,
    mag_filter: Filter,
    mip_filter: MipFilter,
    address: [AddressMode; 3],
    max_anisotropy: u32,
    lod_min_clamp: f32,
    lod_max_clamp: f32,
    compare: Option<CompareFunction>,
    border_color: BorderColor,
    normalized_coordinates: bool,
}

fn parsed<T>(field: &'static str, ordinal: u32, value: Option<T>) -> Result<T, SamplerRefusal> {
    value.ok_or(SamplerRefusal::UnknownOrdinal { field, ordinal })
}

impl SamplerShape {
    /// Parse and check the declaration.
    ///
    /// # Errors
    ///
    /// [`SamplerRefusal`] naming the field that failed.
    pub fn checked(self) -> Result<SamplerState, SamplerRefusal> {
        let min_filter = parsed(
            "min_filter",
            self.min_filter,
            Filter::parse(self.min_filter),
        )?;
        let mag_filter = parsed(
            "mag_filter",
            self.mag_filter,
            Filter::parse(self.mag_filter),
        )?;
        let mip_filter = parsed(
            "mip_filter",
            self.mip_filter,
            MipFilter::parse(self.mip_filter),
        )?;
        let address = [
            parsed(
                "s_address",
                self.s_address,
                AddressMode::parse(self.s_address),
            )?,
            parsed(
                "t_address",
                self.t_address,
                AddressMode::parse(self.t_address),
            )?,
            parsed(
                "r_address",
                self.r_address,
                AddressMode::parse(self.r_address),
            )?,
        ];
        let border_color = parsed(
            "border_color",
            self.border_color,
            BorderColor::parse(self.border_color),
        )?;
        let compare = self
            .compare_enabled
            .then(|| {
                parsed(
                    "compare_function",
                    self.compare_function,
                    CompareFunction::parse(self.compare_function),
                )
            })
            .transpose()?;

        if !self.lod_min_clamp.is_finite()
            || !self.lod_max_clamp.is_finite()
            || self.lod_min_clamp > self.lod_max_clamp
        {
            return Err(SamplerRefusal::BadLodClamp {
                min_bits: self.lod_min_clamp.to_bits(),
                max_bits: self.lod_max_clamp.to_bits(),
            });
        }

        if !self.normalized_coordinates {
            // The restricted mode. Each of these is a rule both APIs place on
            // it, checked once here so two backends cannot each discover a
            // different subset.
            if min_filter != mag_filter {
                return Err(SamplerRefusal::UnnormalizedRestriction { field: "filter" });
            }
            if mip_filter == MipFilter::Linear {
                return Err(SamplerRefusal::UnnormalizedRestriction {
                    field: "mip_filter",
                });
            }
            if self.max_anisotropy > 1 {
                return Err(SamplerRefusal::UnnormalizedRestriction {
                    field: "max_anisotropy",
                });
            }
            if compare.is_some() {
                return Err(SamplerRefusal::UnnormalizedRestriction {
                    field: "compare_function",
                });
            }
            // By index, so the refusal names the axis that broke the rule and
            // not the first axis that happens to hold the same mode.
            for (mode, field) in address.iter().zip(["s_address", "t_address", "r_address"]) {
                if !mode.allows_unnormalized() {
                    return Err(SamplerRefusal::UnnormalizedRestriction { field });
                }
            }
        }

        Ok(SamplerState {
            min_filter,
            mag_filter,
            mip_filter,
            address,
            // One is "off"; the decoder floors it there and nothing below
            // means anything.
            max_anisotropy: self.max_anisotropy.max(1),
            lod_min_clamp: self.lod_min_clamp,
            lod_max_clamp: self.lod_max_clamp,
            compare,
            border_color,
            normalized_coordinates: self.normalized_coordinates,
        })
    }
}

impl SamplerState {
    #[must_use]
    pub const fn min_filter(self) -> Filter {
        self.min_filter
    }

    #[must_use]
    pub const fn mag_filter(self) -> Filter {
        self.mag_filter
    }

    #[must_use]
    pub const fn mip_filter(self) -> MipFilter {
        self.mip_filter
    }

    /// The three axes, in `s`, `t`, `r` order.
    #[must_use]
    pub const fn address(self) -> [AddressMode; 3] {
        self.address
    }

    #[must_use]
    pub const fn max_anisotropy(self) -> u32 {
        self.max_anisotropy
    }

    #[must_use]
    pub const fn lod_clamp(self) -> (f32, f32) {
        (self.lod_min_clamp, self.lod_max_clamp)
    }

    #[must_use]
    pub const fn compare(self) -> Option<CompareFunction> {
        self.compare
    }

    #[must_use]
    pub const fn border_color(self) -> BorderColor {
        self.border_color
    }

    #[must_use]
    pub const fn normalized_coordinates(self) -> bool {
        self.normalized_coordinates
    }

    /// Whether any axis reads a border.
    ///
    /// The question a backend asks before it decides whether the border colour
    /// matters at all — and it is per-sampler in both APIs, which is why the
    /// axes are checked together rather than one at a time.
    #[must_use]
    pub fn uses_border(self) -> bool {
        self.address.iter().any(|m| m.uses_border())
    }

    /// Whether two border modes on different axes disagree about the border.
    ///
    /// `ClampToZero` has a fixed transparent-black border and
    /// `ClampToBorderColor` reads the declared one. A descriptor using both
    /// with a non-transparent border colour is asking for two borders, and
    /// neither API has more than one — so this is the question a backend has
    /// to ask before it picks one.
    #[must_use]
    pub fn border_modes_disagree(self) -> bool {
        let zero = self.address.contains(&AddressMode::ClampToZero);
        let declared = self.address.contains(&AddressMode::ClampToBorderColor);
        zero && declared && self.border_color != BorderColor::TransparentBlack
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;
    use reims_vgpu_observe::Decline;

    fn base() -> SamplerShape {
        SamplerShape {
            min_filter: MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
            mag_filter: MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
            mip_filter: MTL_SAMPLER_MIP_FILTER_LINEAR,
            s_address: MTL_SAMPLER_ADDRESS_MODE_REPEAT,
            t_address: MTL_SAMPLER_ADDRESS_MODE_REPEAT,
            r_address: MTL_SAMPLER_ADDRESS_MODE_REPEAT,
            max_anisotropy: 1,
            lod_min_clamp: 0.0,
            lod_max_clamp: 1000.0,
            compare_function: MTL_COMPARE_FUNCTION_NEVER,
            compare_enabled: false,
            border_color: MTL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK,
            normalized_coordinates: true,
        }
    }

    /// The unnormalized mode's own valid base, so a test mutating one field
    /// sees that field's refusal.
    fn unnormalized() -> SamplerShape {
        SamplerShape {
            mip_filter: MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED,
            s_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            t_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            r_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            normalized_coordinates: false,
            ..base()
        }
    }

    #[test]
    fn every_ordinal_set_is_closed_and_round_trips() {
        for (index, filter) in Filter::ALL.iter().enumerate() {
            assert_eq!(Filter::parse(index as u32), Some(*filter));
        }
        assert_eq!(Filter::parse(2), None);

        for (index, mip) in MipFilter::ALL.iter().enumerate() {
            assert_eq!(MipFilter::parse(index as u32), Some(*mip));
        }
        assert_eq!(MipFilter::parse(3), None);

        for (index, mode) in AddressMode::ALL.iter().enumerate() {
            assert_eq!(AddressMode::parse(index as u32), Some(*mode));
        }
        assert_eq!(AddressMode::parse(6), None);

        for (index, border) in BorderColor::ALL.iter().enumerate() {
            assert_eq!(BorderColor::parse(index as u32), Some(*border));
        }
        assert_eq!(BorderColor::parse(3), None);

        for (index, compare) in CompareFunction::ALL.iter().enumerate() {
            assert_eq!(CompareFunction::parse(index as u32), Some(*compare));
            assert_eq!(compare.ordinal(), index as u32);
        }
        assert_eq!(CompareFunction::parse(8), None);
    }

    #[test]
    fn every_name_in_a_set_is_distinct() {
        for names in [
            Filter::ALL
                .iter()
                .map(|f| f.name())
                .collect::<BTreeSet<_>>(),
            MipFilter::ALL
                .iter()
                .map(|f| f.name())
                .collect::<BTreeSet<_>>(),
            AddressMode::ALL
                .iter()
                .map(|f| f.name())
                .collect::<BTreeSet<_>>(),
            BorderColor::ALL
                .iter()
                .map(|f| f.name())
                .collect::<BTreeSet<_>>(),
            CompareFunction::ALL
                .iter()
                .map(|f| f.name())
                .collect::<BTreeSet<_>>(),
        ] {
            assert!(!names.is_empty());
        }
        assert_eq!(
            AddressMode::ALL
                .iter()
                .map(|m| m.name())
                .collect::<BTreeSet<_>>()
                .len(),
            AddressMode::ALL.len()
        );
    }

    #[test]
    fn an_unknown_ordinal_names_its_field_and_is_never_folded() {
        for (mutate, field, ordinal) in [
            (
                SamplerShape {
                    min_filter: 9,
                    ..base()
                },
                "min_filter",
                9u32,
            ),
            (
                SamplerShape {
                    mip_filter: 9,
                    ..base()
                },
                "mip_filter",
                9,
            ),
            (
                SamplerShape {
                    t_address: 9,
                    ..base()
                },
                "t_address",
                9,
            ),
            (
                SamplerShape {
                    border_color: 9,
                    ..base()
                },
                "border_color",
                9,
            ),
        ] {
            assert_eq!(
                mutate.checked(),
                Err(SamplerRefusal::UnknownOrdinal { field, ordinal })
            );
        }
    }

    #[test]
    fn a_comparison_function_is_read_only_where_one_is_enabled() {
        // An unrecognised function is not an error where no comparison is
        // performed: the field is Metal's default and not a request.
        let disabled = SamplerShape {
            compare_function: 99,
            compare_enabled: false,
            ..base()
        };
        assert_eq!(disabled.checked().expect("no comparison").compare(), None);

        let enabled = SamplerShape {
            compare_function: MTL_COMPARE_FUNCTION_LESS_EQUAL,
            compare_enabled: true,
            ..base()
        };
        assert_eq!(
            enabled.checked().expect("a comparison").compare(),
            Some(CompareFunction::LessEqual)
        );

        let bad = SamplerShape {
            compare_function: 99,
            compare_enabled: true,
            ..base()
        };
        assert!(matches!(
            bad.checked(),
            Err(SamplerRefusal::UnknownOrdinal {
                field: "compare_function",
                ..
            })
        ));
    }

    #[test]
    fn an_inverted_or_absent_lod_clamp_refuses() {
        for (min, max) in [(2.0f32, 1.0f32), (f32::NAN, 1.0), (0.0, f32::INFINITY)] {
            assert!(matches!(
                SamplerShape {
                    lod_min_clamp: min,
                    lod_max_clamp: max,
                    ..base()
                }
                .checked(),
                Err(SamplerRefusal::BadLodClamp { .. })
            ));
        }
        // Equal bounds are a clamp to one level, which is legal.
        assert!(SamplerShape {
            lod_min_clamp: 3.0,
            lod_max_clamp: 3.0,
            ..base()
        }
        .checked()
        .is_ok());
    }

    #[test]
    fn the_unnormalized_mode_refuses_each_restriction_by_the_field_that_broke_it() {
        assert!(unnormalized().checked().is_ok(), "the base is valid");

        for (shape, field) in [
            (
                SamplerShape {
                    min_filter: MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
                    ..unnormalized()
                },
                "filter",
            ),
            (
                SamplerShape {
                    mip_filter: MTL_SAMPLER_MIP_FILTER_LINEAR,
                    ..unnormalized()
                },
                "mip_filter",
            ),
            (
                SamplerShape {
                    max_anisotropy: 4,
                    ..unnormalized()
                },
                "max_anisotropy",
            ),
            (
                SamplerShape {
                    compare_enabled: true,
                    compare_function: MTL_COMPARE_FUNCTION_LESS,
                    ..unnormalized()
                },
                "compare_function",
            ),
            (
                SamplerShape {
                    t_address: MTL_SAMPLER_ADDRESS_MODE_REPEAT,
                    ..unnormalized()
                },
                "t_address",
            ),
        ] {
            assert_eq!(
                shape.checked(),
                Err(SamplerRefusal::UnnormalizedRestriction { field }),
                "{field}"
            );
        }
    }

    #[test]
    fn the_restrictions_apply_only_to_the_unnormalized_mode() {
        // Every one of them is ordinary on a normalized sampler.
        let ordinary = SamplerShape {
            min_filter: MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
            mag_filter: MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
            mip_filter: MTL_SAMPLER_MIP_FILTER_LINEAR,
            max_anisotropy: 16,
            compare_enabled: true,
            compare_function: MTL_COMPARE_FUNCTION_GREATER,
            ..base()
        };
        let state = ordinary.checked().expect("a normalized sampler");
        assert_eq!(state.max_anisotropy(), 16);
        assert!(state.normalized_coordinates());
    }

    #[test]
    fn the_refusal_names_the_axis_that_broke_the_rule_and_not_a_twin() {
        // Two axes hold `ClampToEdge` and the third holds the offending mode.
        // A search by value rather than by position would name the first.
        let shape = SamplerShape {
            r_address: MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT,
            ..unnormalized()
        };
        assert_eq!(
            shape.checked(),
            Err(SamplerRefusal::UnnormalizedRestriction { field: "r_address" })
        );
    }

    #[test]
    fn anisotropy_is_floored_at_one_because_below_it_means_nothing() {
        let state = SamplerShape {
            max_anisotropy: 0,
            ..base()
        }
        .checked()
        .expect("a sampler");
        assert_eq!(state.max_anisotropy(), 1);
    }

    #[test]
    fn a_border_is_used_only_where_an_axis_reads_one() {
        let none = base().checked().expect("repeat on every axis");
        assert!(!none.uses_border());

        for mode in [
            MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO,
            MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER_COLOR,
        ] {
            let state = SamplerShape {
                t_address: mode,
                ..base()
            }
            .checked()
            .expect("a sampler");
            assert!(state.uses_border());
        }
    }

    #[test]
    fn two_border_modes_disagree_only_when_the_declared_border_is_not_transparent() {
        // `ClampToZero` is a fixed transparent-black border and
        // `ClampToBorderColor` reads the declared one. Together with an opaque
        // declaration that is two borders, and neither API has two.
        let clash = SamplerShape {
            s_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO,
            t_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER_COLOR,
            border_color: MTL_SAMPLER_BORDER_COLOR_OPAQUE_WHITE,
            ..base()
        }
        .checked()
        .expect("a declaration the guest API admits");
        assert!(clash.border_modes_disagree());

        // The same pair with a transparent declaration is one border twice.
        let agreed = SamplerShape {
            s_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO,
            t_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER_COLOR,
            border_color: MTL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK,
            ..base()
        }
        .checked()
        .expect("a sampler");
        assert!(!agreed.border_modes_disagree());

        // And one border mode alone never disagrees, whatever the colour.
        let alone = SamplerShape {
            s_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER_COLOR,
            border_color: MTL_SAMPLER_BORDER_COLOR_OPAQUE_BLACK,
            ..base()
        }
        .checked()
        .expect("a sampler");
        assert!(!alone.border_modes_disagree());
    }

    #[test]
    fn every_refusal_names_itself() {
        let refusals = [
            SamplerRefusal::UnknownOrdinal {
                field: "min_filter",
                ordinal: 9,
            },
            SamplerRefusal::UnnormalizedRestriction { field: "filter" },
            SamplerRefusal::BadLodClamp {
                min_bits: 1,
                max_bits: 0,
            },
        ];
        let slugs: BTreeSet<&str> = refusals.iter().map(Decline::slug).collect();
        assert_eq!(slugs.len(), refusals.len());
        for refusal in refusals {
            assert!(refusal.slug().starts_with("sampler_"));
            assert!(!refusal.fields().is_empty());
        }
    }
}
