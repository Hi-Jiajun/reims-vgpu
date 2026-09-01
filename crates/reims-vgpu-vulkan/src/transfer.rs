//! What a resolved transfer becomes here: the exact native copies, planned
//! against the resources the names resolve to and nothing else.
//!
//! # A pitch is bytes to the guest and texels to Vulkan
//!
//! `MTLBlitCommandEncoder` describes a linear image in a buffer by its
//! `bytesPerRow` and `bytesPerImage`. `VkBufferImageCopy` describes the same
//! thing by `bufferRowLength` and `bufferImageHeight`, which are **texels**.
//! The conversion is the format's, and it is exact or it is a refusal: a
//! `bytesPerRow` that is not a whole number of blocks describes a row this
//! copy cannot express, and rounding it either way copies the wrong bytes into
//! a texture that then samples plausibly wrong.
//!
//! Block-compressed formats make the difference visible — a BC row is
//! `blocks_across(width) * block_bytes`, so the texel row length is the block
//! count times the block width, and a conversion that divided bytes by a
//! bytes-per-pixel that does not exist for these formats would be off by the
//! block size.
//!
//! # A fill is four bytes at a time or it is refused
//!
//! `vkCmdFillBuffer` requires a four-byte-aligned offset and size, and the
//! guest's `fillBuffer:range:` does not. A ragged range therefore cannot
//! become one fill command, and it is refused by name rather than rounded —
//! rounding out writes bytes outside the range the guest named, and rounding
//! in leaves bytes inside it unwritten. Closing it needs a staging copy, which
//! has no alignment rule at all; [`crate::placement`] already routes one, and
//! this refusal is what says the route is needed.
//!
//! A one-byte pattern is replicated into the four, which is what the guest's
//! own `fillBuffer:range:value:` means.
//!
//! # Planned, not recorded
//!
//! Every function here produces commands and issues none, so the pitch
//! arithmetic, the region expansion and the refusals are tested with no GPU.
//! The layout each image has to be in for these copies is
//! [`crate::layout`]'s answer and is deliberately not decided here: a transfer
//! that also chose layouts would be a second place image state is tracked.

use ash::vk;
use reims_vgpu_core::blit::{BlitOp, ImagePitch, Origin3, Size3, TexturePoint};
use reims_vgpu_core::pixel_format::block_geometry;
use reims_vgpu_core::texture_shape::Texture;

use crate::resident::{Miss, Residency};
use crate::view::aspect;

/// Why a resolved transfer cannot be recorded on this host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A name in the transfer did not resolve.
    Unresolved { miss: Miss },
    /// `bytesPerRow` is not a whole number of the format's blocks, so the row
    /// cannot be expressed in texels.
    RowPitchNotWholeBlocks {
        bytes_per_row: u64,
        block_bytes: u32,
    },
    /// `bytesPerImage` is not a whole number of rows.
    ImagePitchNotWholeRows {
        bytes_per_image: u64,
        bytes_per_row: u64,
    },
    /// The guest's format has no block geometry this build knows, so no pitch
    /// conversion exists for it.
    UnknownFormatGeometry { format: u16 },
    /// A fill whose offset or length is not four-byte aligned, and whose
    /// pattern is four bytes wide.
    ///
    /// The byte-wide pattern has a ragged form — see [`plan_fill`] — because
    /// every byte of the range takes the same value and the phase of the
    /// pattern therefore does not exist. A four-byte pattern has a phase, and
    /// which byte of it lands on a range that does not start at a multiple of
    /// four is not a term this wire has established. Refused rather than
    /// guessed: a wrong phase writes the right number of bytes in the wrong
    /// order, everywhere, and nothing downstream can see it.
    RaggedPatternFill { offset: u64, length: u64 },
    /// The scratch memory a ragged fill needs was not available.
    NoStaging { refusal: crate::staging::Refusal },
    /// A fill or copy of no bytes.
    EmptyRange,
    /// A dimension larger than the 32-bit fields a native copy carries.
    ///
    /// The guest's own values are 64-bit; this is where that stops being true,
    /// and it is checked rather than truncated because a truncated extent is a
    /// copy that succeeds and moves the wrong bytes.
    ExtentTooLarge { axis: &'static str, value: u64 },
    /// A blit-encoder operation that is not a copy, and so has no native
    /// transfer form at all.
    ///
    /// Named rather than absent: a caller reaching this has routed the
    /// operation to the wrong planner, and the refusal says which one it is.
    NotACopy { op: &'static str },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Unresolved { .. } => "vk_transfer_unresolved",
            Self::RowPitchNotWholeBlocks { .. } => "vk_transfer_row_pitch_not_whole_blocks",
            Self::ImagePitchNotWholeRows { .. } => "vk_transfer_image_pitch_not_whole_rows",
            Self::UnknownFormatGeometry { .. } => "vk_transfer_unknown_format_geometry",
            Self::RaggedPatternFill { .. } => "vk_transfer_ragged_pattern_fill",
            Self::NoStaging { .. } => "vk_transfer_no_staging",
            Self::EmptyRange => "vk_transfer_empty_range",
            Self::ExtentTooLarge { .. } => "vk_transfer_extent_too_large",
            Self::NotACopy { .. } => "vk_transfer_not_a_copy",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unresolved { miss } => write!(f, "{} {miss}", self.slug()),
            Self::RowPitchNotWholeBlocks {
                bytes_per_row,
                block_bytes,
            } => write!(
                f,
                "{} bytes_per_row={bytes_per_row} block_bytes={block_bytes}",
                self.slug()
            ),
            Self::ImagePitchNotWholeRows {
                bytes_per_image,
                bytes_per_row,
            } => write!(
                f,
                "{} bytes_per_image={bytes_per_image} bytes_per_row={bytes_per_row}",
                self.slug()
            ),
            Self::UnknownFormatGeometry { format } => {
                write!(f, "{} format={format}", self.slug())
            }
            Self::RaggedPatternFill { offset, length } => {
                write!(f, "{} offset={offset} length={length}", self.slug())
            }
            Self::NoStaging { refusal } => write!(f, "{} {refusal}", self.slug()),
            Self::EmptyRange => f.write_str(self.slug()),
            Self::ExtentTooLarge { axis, value } => {
                write!(f, "{} axis={axis} value={value}", self.slug())
            }
            Self::NotACopy { op } => write!(f, "{} op={op}", self.slug()),
        }
    }
}

/// One native transfer command, with the handles it names already resolved.
///
/// Not `PartialEq`: ash's copy structures are not, and wrapping nine `u32`
/// fields per region to make them so would be a second vocabulary for
/// `VkBufferImageCopy` whose only reader is a test. The tests match and assert
/// fields instead.
#[derive(Clone, Debug)]
pub enum Command {
    CopyBuffer {
        source: vk::Buffer,
        dest: vk::Buffer,
        regions: Vec<vk::BufferCopy>,
    },
    CopyBufferToImage {
        source: vk::Buffer,
        dest: vk::Image,
        regions: Vec<vk::BufferImageCopy>,
    },
    CopyImageToBuffer {
        source: vk::Image,
        dest: vk::Buffer,
        regions: Vec<vk::BufferImageCopy>,
    },
    CopyImage {
        source: vk::Image,
        dest: vk::Image,
        regions: Vec<vk::ImageCopy>,
    },
}

/// A dimension the guest gave in 64 bits, as the 32 a native copy carries.
fn narrow(axis: &'static str, value: u64) -> Result<u32, Refusal> {
    u32::try_from(value).map_err(|_| Refusal::ExtentTooLarge { axis, value })
}

fn extent(size: Size3) -> Result<vk::Extent3D, Refusal> {
    Ok(vk::Extent3D {
        width: narrow("width", size.width)?,
        height: narrow("height", size.height)?,
        depth: narrow("depth", size.depth)?,
    })
}

fn offset(origin: Origin3) -> Result<vk::Offset3D, Refusal> {
    Ok(vk::Offset3D {
        x: i32::try_from(origin.x).map_err(|_| Refusal::ExtentTooLarge {
            axis: "origin_x",
            value: origin.x,
        })?,
        y: i32::try_from(origin.y).map_err(|_| Refusal::ExtentTooLarge {
            axis: "origin_y",
            value: origin.y,
        })?,
        z: i32::try_from(origin.z).map_err(|_| Refusal::ExtentTooLarge {
            axis: "origin_z",
            value: origin.z,
        })?,
    })
}

/// The subresource layers one texture endpoint names.
fn layers(texture: Texture, point: TexturePoint) -> Result<vk::ImageSubresourceLayers, Refusal> {
    Ok(vk::ImageSubresourceLayers {
        aspect_mask: aspect(texture.pixel_format()),
        mip_level: u32::from(point.level),
        base_array_layer: u32::from(point.slice),
        layer_count: 1,
    })
}

/// The buffer-side pitch of a linear image, in texels.
///
/// Zero is the guest saying "tightly packed", and Vulkan spells that with zero
/// too, so it passes through rather than being computed — computing it would
/// produce the same number for a tightly packed copy and a different one for a
/// copy whose extent is not the whole row.
///
/// # Errors
///
/// [`Refusal`] when the byte pitch is not a whole number of blocks or rows, or
/// when this build has no geometry for the format.
pub fn texel_pitch(texture: Texture, pitch: ImagePitch) -> Result<(u32, u32), Refusal> {
    let format = texture.pixel_format();
    let block = block_geometry(format).ok_or(Refusal::UnknownFormatGeometry { format })?;
    let block_bytes = u64::from(block.bytes);

    let row_length = if pitch.bytes_per_row == 0 {
        0
    } else {
        if block_bytes == 0 || !pitch.bytes_per_row.is_multiple_of(block_bytes) {
            return Err(Refusal::RowPitchNotWholeBlocks {
                bytes_per_row: pitch.bytes_per_row,
                block_bytes: block.bytes,
            });
        }
        // Blocks across, times the texels a block spans: for an uncompressed
        // format the block is 1x1 and this is the byte count over the texel
        // size, which is the same expression.
        narrow(
            "row_length",
            (pitch.bytes_per_row / block_bytes) * u64::from(block.width),
        )?
    };

    let image_height = if pitch.bytes_per_image == 0 || pitch.bytes_per_row == 0 {
        0
    } else {
        if !pitch.bytes_per_image.is_multiple_of(pitch.bytes_per_row) {
            return Err(Refusal::ImagePitchNotWholeRows {
                bytes_per_image: pitch.bytes_per_image,
                bytes_per_row: pitch.bytes_per_row,
            });
        }
        narrow(
            "image_height",
            (pitch.bytes_per_image / pitch.bytes_per_row) * u64::from(block.height),
        )?
    };

    Ok((row_length, image_height))
}

/// The four-byte word a fill pattern repeats.
///
/// A one-byte pattern is replicated, which is what `fillBuffer:range:value:`
/// means: every byte of the range takes the value.
#[must_use]
pub const fn fill_word(pattern: reims_vgpu_core::blit::FillPattern) -> u32 {
    match pattern {
        reims_vgpu_core::blit::FillPattern::Byte(byte) => u32::from_ne_bytes([byte; 4]),
        reims_vgpu_core::blit::FillPattern::Pattern4(word) => word,
    }
}

fn resolved<T>(result: Result<T, Miss>) -> Result<T, Refusal> {
    result.map_err(|miss| Refusal::Unresolved { miss })
}

/// Plan the native commands one resolved transfer becomes.
///
/// # Errors
///
/// [`Refusal`] naming the one thing that could not be expressed.
pub fn plan(op: &BlitOp, residency: &Residency) -> Result<Command, Refusal> {
    match *op {
        BlitOp::BufferToBuffer {
            source,
            source_offset,
            dest,
            dest_offset,
            size,
        } => {
            if size == 0 {
                return Err(Refusal::EmptyRange);
            }
            Ok(Command::CopyBuffer {
                source: resolved(residency.buffer(source))?.buffer,
                dest: resolved(residency.buffer(dest))?.buffer,
                regions: vec![vk::BufferCopy {
                    src_offset: source_offset,
                    dst_offset: dest_offset,
                    size,
                }],
            })
        }
        BlitOp::BufferToTexture {
            source,
            source_offset,
            source_pitch,
            size,
            dest,
            options: _,
        } => {
            let buffer = resolved(residency.buffer(source))?.buffer;
            let image = resolved(residency.image(dest.texture))?;
            Ok(Command::CopyBufferToImage {
                source: buffer,
                dest: image.image,
                regions: vec![buffer_image_region(
                    image.texture,
                    source_offset,
                    source_pitch,
                    dest,
                    size,
                )?],
            })
        }
        BlitOp::TextureToBuffer {
            source,
            size,
            dest,
            dest_offset,
            dest_pitch,
            options: _,
        } => {
            let image = resolved(residency.image(source.texture))?;
            let buffer = resolved(residency.buffer(dest))?.buffer;
            Ok(Command::CopyImageToBuffer {
                source: image.image,
                dest: buffer,
                regions: vec![buffer_image_region(
                    image.texture,
                    dest_offset,
                    dest_pitch,
                    source,
                    size,
                )?],
            })
        }
        BlitOp::TextureRegion {
            source,
            dest,
            size,
            options: _,
        } => {
            let from = resolved(residency.image(source.texture))?;
            let to = resolved(residency.image(dest.texture))?;
            Ok(Command::CopyImage {
                source: from.image,
                dest: to.image,
                regions: vec![vk::ImageCopy {
                    src_subresource: layers(from.texture, source)?,
                    src_offset: offset(source.origin)?,
                    dst_subresource: layers(to.texture, dest)?,
                    dst_offset: offset(dest.origin)?,
                    extent: extent(size)?,
                }],
            })
        }
        BlitOp::TextureSlices { source, dest } => {
            let from = resolved(residency.image(source.texture))?;
            let to = resolved(residency.image(dest.texture))?;
            let mut regions = Vec::with_capacity(usize::from(source.level_count));
            for level in 0..u32::from(source.level_count) {
                // One region per level and not per slice: `layerCount` covers
                // the slice span, and the extent is the level's own — a mip
                // chain copied with level zero's extent reads past every level
                // below it.
                let source_extent = from
                    .texture
                    .level_extent(u32::from(source.base_level) + level)
                    .ok_or(Refusal::ExtentTooLarge {
                        axis: "level",
                        value: u64::from(source.base_level) + u64::from(level),
                    })?;
                regions.push(vk::ImageCopy {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: aspect(from.texture.pixel_format()),
                        mip_level: u32::from(source.base_level) + level,
                        base_array_layer: u32::from(source.base_slice),
                        layer_count: u32::from(source.slice_count),
                    },
                    src_offset: vk::Offset3D::default(),
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: aspect(to.texture.pixel_format()),
                        mip_level: u32::from(dest.base_level) + level,
                        base_array_layer: u32::from(dest.base_slice),
                        layer_count: u32::from(dest.slice_count),
                    },
                    dst_offset: vk::Offset3D::default(),
                    extent: vk::Extent3D {
                        width: source_extent.x,
                        height: source_extent.y,
                        depth: source_extent.z,
                    },
                });
            }
            if regions.is_empty() {
                return Err(Refusal::EmptyRange);
            }
            Ok(Command::CopyImage {
                source: from.image,
                dest: to.image,
                regions,
            })
        }
        // A fill is not a copy, and a ragged one is not even one command. See
        // [`plan_fill`], which is the only thing that can answer it, because
        // it needs the scratch memory this signature has no access to.
        BlitOp::FillBuffer { .. } => Err(Refusal::NotACopy { op: "fill_buffer" }),
        // A filtered reduction with a barrier between every pair of levels,
        // not a copy: see [`crate::mipmap`]. Refused here rather than absorbed
        // into one of the copies above, because a mipmap generation recorded
        // as a copy would produce level one and leave the rest of the chain
        // undefined. The name is still resolved first, so a generation naming
        // nothing is that refusal and not this one.
        BlitOp::GenerateMipmaps { texture } => {
            resolved(residency.image(texture))?;
            Err(Refusal::NotACopy {
                op: "generate_mipmaps",
            })
        }
    }
}

/// The interior of a fill, as `vkCmdFillBuffer` takes it.
///
/// Its own type rather than a [`Command`], so a [`FillPlan`] stays comparable:
/// a plan whose parts cannot be compared is one whose split cannot be asserted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FillRange {
    pub offset: u64,
    pub size: u64,
    pub data: u32,
}

/// The bytes a ragged fill's edge needs, and where they go.
///
/// The caller writes `byte` into every one of `length` bytes at the window,
/// flushes if the mapping is not coherent, and copies the window into the
/// destination buffer at `dest_offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StagedEdge {
    pub window: crate::staging::Window,
    pub byte: u8,
    pub dest_offset: u64,
    pub length: u64,
}

/// A fill, split into the part a native fill command can do and the edges it
/// cannot.
///
/// `head` and `tail` are at most three bytes each, so the scratch this costs
/// is six bytes whatever the size of the fill — the middle is still one
/// `vkCmdFillBuffer` over everything else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "a fill plan whose edges are not written leaves the range partly unfilled"]
pub struct FillPlan {
    pub dest: vk::Buffer,
    /// The unaligned bytes before the first four-byte boundary in the range.
    pub head: Option<StagedEdge>,
    /// The four-byte-aligned interior. `None` only for a range too short to
    /// contain one.
    pub middle: Option<FillRange>,
    /// The unaligned bytes after the last four-byte boundary in the range.
    pub tail: Option<StagedEdge>,
}

/// Plan a buffer fill, using scratch memory for the edges a native fill cannot
/// reach.
///
/// `vkCmdFillBuffer` requires a four-byte-aligned offset and size, and
/// `fillBuffer:range:` does not. Refusing the whole fill for that would lose a
/// legal guest command; rounding out would write bytes outside the range the
/// guest named, and rounding in would leave bytes inside it unwritten. So the
/// range is split: the interior is one fill command, and the at-most-three
/// bytes at each end are staged and copied. A copy has no alignment rule at
/// all, which is exactly why the edges go through one.
///
/// # Errors
///
/// [`Refusal`]. Scratch is taken in one allocation covering both edges, so a
/// refusal leaves the arena as it was rather than stranding a head window
/// whose tail could not be found.
pub fn plan_fill(
    span: reims_vgpu_core::blit::BufferSpan,
    pattern: reims_vgpu_core::blit::FillPattern,
    residency: &Residency,
    arena: &mut crate::staging::Arena,
) -> Result<FillPlan, Refusal> {
    use reims_vgpu_core::blit::FillPattern;

    if span.length == 0 {
        return Err(Refusal::EmptyRange);
    }
    let dest = resolved(residency.buffer(span.buffer))?.buffer;

    // The bytes before the first four-byte boundary inside the range, and
    // after the last one. Both are zero for an aligned range, which is the
    // common case and costs no scratch at all.
    let head_length = (4 - span.offset % 4) % 4;
    let head_length = head_length.min(span.length);
    let remaining = span.length - head_length;
    let middle_length = remaining & !3;
    let tail_length = remaining & 3;

    if head_length == 0 && tail_length == 0 {
        return Ok(FillPlan {
            dest,
            head: None,
            middle: Some(FillRange {
                offset: span.offset,
                size: span.length,
                data: fill_word(pattern),
            }),
            tail: None,
        });
    }

    // A byte pattern has no phase: every byte of the range takes the value, so
    // splitting the range changes nothing. A four-byte pattern does have one,
    // and which of its bytes lands on an unaligned start is not established.
    let FillPattern::Byte(byte) = pattern else {
        return Err(Refusal::RaggedPatternFill {
            offset: span.offset,
            length: span.length,
        });
    };

    // One allocation for both edges, so there is one failure point and no
    // stranded window on it.
    let scratch = arena
        .allocate(head_length + tail_length, 1)
        .map_err(|refusal| Refusal::NoStaging { refusal })?;

    let head = (head_length > 0).then_some(StagedEdge {
        window: crate::staging::Window {
            chunk: scratch.chunk,
            offset: scratch.offset,
            size: head_length,
        },
        byte,
        dest_offset: span.offset,
        length: head_length,
    });
    let tail = (tail_length > 0).then_some(StagedEdge {
        window: crate::staging::Window {
            chunk: scratch.chunk,
            offset: scratch.offset + head_length,
            size: tail_length,
        },
        byte,
        dest_offset: span.offset + head_length + middle_length,
        length: tail_length,
    });
    let middle = (middle_length > 0).then_some(FillRange {
        offset: span.offset + head_length,
        size: middle_length,
        data: fill_word(pattern),
    });

    Ok(FillPlan {
        dest,
        head,
        middle,
        tail,
    })
}

/// One `VkBufferImageCopy`, whichever direction it goes.
///
/// Shared because the structure is the same both ways and the two directions
/// disagreeing about the pitch conversion would be a download that does not
/// match the upload that produced it.
fn buffer_image_region(
    texture: Texture,
    buffer_offset: u64,
    pitch: ImagePitch,
    point: TexturePoint,
    size: Size3,
) -> Result<vk::BufferImageCopy, Refusal> {
    let (row_length, image_height) = texel_pitch(texture, pitch)?;
    Ok(vk::BufferImageCopy {
        buffer_offset,
        buffer_row_length: row_length,
        buffer_image_height: image_height,
        image_subresource: layers(texture, point)?,
        image_offset: offset(point.origin)?,
        image_extent: extent(size)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use reims_vgpu_core::blit::{BufferSpan, FillPattern, TextureSpan};
    use reims_vgpu_core::identity::{
        DeviceEpoch, ObjectListRef, ResourceId, SessionGeneration, SlotGeneration, TimelinePoint,
    };
    use reims_vgpu_core::pixel_format::{
        MTL_FORMAT_BC3_RGBA, MTL_FORMAT_DEPTH32_FLOAT, MTL_FORMAT_RGBA8_UNORM,
    };
    use reims_vgpu_core::retire::{Lifetime, NativeRetirement};
    use reims_vgpu_core::texture_shape::{TextureKind, TextureShape, TextureUsage};
    use std::collections::BTreeSet;

    use crate::buffer::{BufferPlan, EVERY_CLASS};
    use crate::image::ImagePlan;
    use crate::resident::{Native, NativeBuffer, NativeImage};

    const BUFFER_A: u32 = 1;
    const BUFFER_B: u32 = 2;
    const IMAGE_A: u32 = 3;
    const IMAGE_B: u32 = 4;

    fn id(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn texture(format: u16, width: u32, height: u32, levels: u32, layers: u32) -> Texture {
        TextureShape {
            kind: if layers > 1 {
                TextureKind::D2Array.ordinal()
            } else {
                TextureKind::D2.ordinal()
            },
            width,
            height,
            depth: 1,
            mipmap_level_count: levels,
            sample_count: 1,
            array_length: layers,
            pixel_format: format,
            usage: TextureUsage::SHADER_READ,
        }
        .checked()
        .expect("a valid declaration")
    }

    fn native_image(handle: u64, texture: Texture) -> Native {
        Native::Image(NativeImage {
            texture,
            image: vk::Image::from_raw(handle),
            memory: vk::DeviceMemory::from_raw(handle),
            plan: ImagePlan {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UNORM,
                extent: vk::Extent3D {
                    width: texture.extent().x,
                    height: texture.extent().y,
                    depth: 1,
                },
                mip_levels: texture.mip_levels(),
                array_layers: texture.layers(),
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::SAMPLED,
                flags: vk::ImageCreateFlags::empty(),
            },
            sampled: vk::ImageView::from_raw(handle),
            attachments: Vec::new(),
        })
    }

    fn native_buffer(handle: u64) -> Native {
        Native::Buffer(NativeBuffer {
            buffer: vk::Buffer::from_raw(handle),
            memory: vk::DeviceMemory::from_raw(handle),
            plan: BufferPlan {
                size: 1 << 20,
                usage: EVERY_CLASS,
                aliased: false,
            },
        })
    }

    /// Two buffers and two textures, published and resolvable.
    fn populated() -> Residency {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        let lifetime = Lifetime::new(SessionGeneration::FIRST, DeviceEpoch::FIRST);
        for (slot, native) in [
            (BUFFER_A, native_buffer(0xB1)),
            (BUFFER_B, native_buffer(0xB2)),
            (
                IMAGE_A,
                native_image(0x1A, texture(MTL_FORMAT_RGBA8_UNORM, 64, 32, 4, 3)),
            ),
            (
                IMAGE_B,
                native_image(0x1B, texture(MTL_FORMAT_RGBA8_UNORM, 64, 32, 4, 3)),
            ),
        ] {
            residency
                .publish(id(slot), lifetime, native, &mut retire)
                .unwrap_or_else(|(_, e)| panic!("{e}"));
        }
        assert_eq!(retire.outstanding(), 0);
        residency
    }

    fn point(slot: u32, level: u16, slice: u16) -> TexturePoint {
        TexturePoint {
            texture: id(slot),
            slice,
            level,
            origin: Origin3 { x: 0, y: 0, z: 0 },
        }
    }

    fn size(width: u64, height: u64) -> Size3 {
        Size3 {
            width,
            height,
            depth: 1,
        }
    }

    #[test]
    fn a_buffer_copy_carries_both_offsets_and_the_size() {
        let residency = populated();
        let planned = plan(
            &BlitOp::BufferToBuffer {
                source: id(BUFFER_A),
                source_offset: 16,
                dest: id(BUFFER_B),
                dest_offset: 64,
                size: 256,
            },
            &residency,
        )
        .expect("plannable");
        let Command::CopyBuffer {
            source,
            dest,
            regions,
        } = planned
        else {
            panic!("a buffer copy");
        };
        assert_eq!(source.as_raw(), 0xB1);
        assert_eq!(dest.as_raw(), 0xB2);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].src_offset, 16);
        assert_eq!(regions[0].dst_offset, 64);
        assert_eq!(regions[0].size, 256);
    }

    #[test]
    fn an_unresolved_name_refuses_with_the_miss_that_produced_it() {
        let residency = populated();
        let refusal = plan(
            &BlitOp::BufferToBuffer {
                source: id(99),
                source_offset: 0,
                dest: id(BUFFER_B),
                dest_offset: 0,
                size: 4,
            },
            &residency,
        )
        .expect_err("no such slot");
        assert_eq!(
            refusal,
            Refusal::Unresolved {
                miss: Miss::Unknown {
                    slot: ObjectListRef(99)
                }
            }
        );

        // And the kind confusion, which is the one a slot-only check misses.
        let wrong = plan(
            &BlitOp::BufferToBuffer {
                source: id(IMAGE_A),
                source_offset: 0,
                dest: id(BUFFER_B),
                dest_offset: 0,
                size: 4,
            },
            &residency,
        )
        .expect_err("a texture is not a buffer");
        assert!(matches!(
            wrong,
            Refusal::Unresolved {
                miss: Miss::WrongKind { .. }
            }
        ));
    }

    #[test]
    fn a_byte_pitch_becomes_a_texel_pitch_through_the_format() {
        // RGBA8: a 1x1 block of four bytes, so 256 bytes is 64 texels.
        let flat = texture(MTL_FORMAT_RGBA8_UNORM, 64, 32, 1, 1);
        assert_eq!(
            texel_pitch(
                flat,
                ImagePitch {
                    bytes_per_row: 256,
                    bytes_per_image: 256 * 32,
                }
            ),
            Ok((64, 32))
        );
    }

    #[test]
    fn a_compressed_row_is_blocks_across_times_the_block_width() {
        // BC3: a 4x4 block of sixteen bytes. A 64-texel row is sixteen blocks
        // and 256 bytes, so the same byte pitch is *64* texels here too — and
        // dividing bytes by a bytes-per-pixel that does not exist for this
        // format would have produced sixteen.
        let compressed = texture(MTL_FORMAT_BC3_RGBA, 64, 32, 1, 1);
        assert_eq!(
            texel_pitch(
                compressed,
                ImagePitch {
                    bytes_per_row: 256,
                    bytes_per_image: 256 * 8,
                }
            ),
            // Eight rows of blocks, each spanning four texels vertically.
            Ok((64, 32))
        );
    }

    #[test]
    fn a_tightly_packed_pitch_passes_through_as_zero() {
        let flat = texture(MTL_FORMAT_RGBA8_UNORM, 64, 32, 1, 1);
        assert_eq!(
            texel_pitch(
                flat,
                ImagePitch {
                    bytes_per_row: 0,
                    bytes_per_image: 0,
                }
            ),
            Ok((0, 0))
        );
        // An image pitch with no row pitch has nothing to be a multiple of, so
        // it is tight too rather than a division by zero.
        assert_eq!(
            texel_pitch(
                flat,
                ImagePitch {
                    bytes_per_row: 0,
                    bytes_per_image: 1024,
                }
            ),
            Ok((0, 0))
        );
    }

    #[test]
    fn a_pitch_that_is_not_whole_blocks_refuses_rather_than_rounding() {
        let flat = texture(MTL_FORMAT_RGBA8_UNORM, 64, 32, 1, 1);
        assert_eq!(
            texel_pitch(
                flat,
                ImagePitch {
                    bytes_per_row: 258,
                    bytes_per_image: 0,
                }
            ),
            Err(Refusal::RowPitchNotWholeBlocks {
                bytes_per_row: 258,
                block_bytes: 4,
            })
        );
        assert_eq!(
            texel_pitch(
                flat,
                ImagePitch {
                    bytes_per_row: 256,
                    bytes_per_image: 300,
                }
            ),
            Err(Refusal::ImagePitchNotWholeRows {
                bytes_per_image: 300,
                bytes_per_row: 256,
            })
        );
    }

    #[test]
    fn a_format_with_no_known_geometry_refuses_by_name() {
        let unknown = TextureShape {
            kind: TextureKind::D2.ordinal(),
            width: 4,
            height: 4,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            // Not a format this build has a block for.
            pixel_format: 0xFFFF,
            usage: TextureUsage::SHADER_READ,
        }
        .checked()
        .expect("a declaration");
        assert_eq!(
            texel_pitch(
                unknown,
                ImagePitch {
                    bytes_per_row: 16,
                    bytes_per_image: 0,
                }
            ),
            Err(Refusal::UnknownFormatGeometry { format: 0xFFFF })
        );
    }

    #[test]
    fn both_directions_use_one_pitch_conversion() {
        let residency = populated();
        let pitch = ImagePitch {
            bytes_per_row: 256,
            bytes_per_image: 256 * 32,
        };
        let up = plan(
            &BlitOp::BufferToTexture {
                source: id(BUFFER_A),
                source_offset: 128,
                source_pitch: pitch,
                size: size(64, 32),
                dest: point(IMAGE_A, 1, 2),
                options: Default::default(),
            },
            &residency,
        )
        .expect("plannable");
        let down = plan(
            &BlitOp::TextureToBuffer {
                source: point(IMAGE_A, 1, 2),
                size: size(64, 32),
                dest: id(BUFFER_A),
                dest_offset: 128,
                dest_pitch: pitch,
                options: Default::default(),
            },
            &residency,
        )
        .expect("plannable");

        let (
            Command::CopyBufferToImage { regions: u, .. },
            Command::CopyImageToBuffer { regions: d, .. },
        ) = (up, down)
        else {
            panic!("one of each direction");
        };
        assert_eq!(u.len(), 1);
        assert_eq!(d.len(), 1);
        // The same region both ways: a download that did not match the upload
        // that produced it is the failure this shares one function for.
        assert_eq!(u[0].buffer_offset, d[0].buffer_offset);
        assert_eq!(u[0].buffer_row_length, d[0].buffer_row_length);
        assert_eq!(u[0].buffer_image_height, d[0].buffer_image_height);
        assert_eq!(u[0].image_subresource.mip_level, 1);
        assert_eq!(u[0].image_subresource.base_array_layer, 2);
        assert_eq!(u[0].image_subresource.layer_count, 1);
        assert_eq!(
            u[0].image_subresource.aspect_mask,
            vk::ImageAspectFlags::COLOR
        );
        assert_eq!(u[0].image_extent.width, 64);
        assert_eq!(u[0].buffer_row_length, 64);
    }

    #[test]
    fn a_depth_texture_transfers_on_its_own_aspect() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(
                id(IMAGE_A),
                Lifetime::new(SessionGeneration::FIRST, DeviceEpoch::FIRST),
                native_image(0x1A, texture(MTL_FORMAT_DEPTH32_FLOAT, 8, 8, 1, 1)),
                &mut retire,
            )
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        residency
            .publish(
                id(BUFFER_A),
                Lifetime::new(SessionGeneration::FIRST, DeviceEpoch::FIRST),
                native_buffer(0xB1),
                &mut retire,
            )
            .unwrap_or_else(|(_, e)| panic!("{e}"));

        let planned = plan(
            &BlitOp::TextureToBuffer {
                source: point(IMAGE_A, 0, 0),
                size: size(8, 8),
                dest: id(BUFFER_A),
                dest_offset: 0,
                dest_pitch: ImagePitch {
                    bytes_per_row: 32,
                    bytes_per_image: 0,
                },
                options: Default::default(),
            },
            &residency,
        )
        .expect("plannable");
        let Command::CopyImageToBuffer { regions, .. } = planned else {
            panic!("a download");
        };
        assert_eq!(
            regions[0].image_subresource.aspect_mask,
            vk::ImageAspectFlags::DEPTH
        );
    }

    #[test]
    fn a_slice_span_copies_one_region_per_level_at_that_levels_extent() {
        let residency = populated();
        let planned = plan(
            &BlitOp::TextureSlices {
                source: TextureSpan {
                    texture: id(IMAGE_A),
                    base_slice: 0,
                    base_level: 1,
                    slice_count: 3,
                    level_count: 3,
                },
                dest: TextureSpan {
                    texture: id(IMAGE_B),
                    base_slice: 0,
                    base_level: 0,
                    slice_count: 3,
                    level_count: 3,
                },
            },
            &residency,
        )
        .expect("plannable");
        let Command::CopyImage { regions, .. } = planned else {
            panic!("an image copy");
        };
        assert_eq!(regions.len(), 3);
        // 64x32 at level 0, so levels one through three are 32x16, 16x8, 8x4.
        // Copying all three with level zero's extent reads past every level
        // below the first.
        let extents: Vec<(u32, u32)> = regions
            .iter()
            .map(|r| (r.extent.width, r.extent.height))
            .collect();
        assert_eq!(extents, [(32, 16), (16, 8), (8, 4)]);
        for (index, region) in regions.iter().enumerate() {
            assert_eq!(region.src_subresource.mip_level, 1 + index as u32);
            assert_eq!(region.dst_subresource.mip_level, index as u32);
            // The slice span is the layer count, not a region each.
            assert_eq!(region.src_subresource.layer_count, 3);
            assert_eq!(region.dst_subresource.layer_count, 3);
        }
    }

    #[test]
    fn an_empty_transfer_is_refused_rather_than_recorded() {
        let residency = populated();
        assert_eq!(
            plan(
                &BlitOp::BufferToBuffer {
                    source: id(BUFFER_A),
                    source_offset: 0,
                    dest: id(BUFFER_B),
                    dest_offset: 0,
                    size: 0,
                },
                &residency,
            )
            .err(),
            Some(Refusal::EmptyRange)
        );
        assert_eq!(
            plan(
                &BlitOp::TextureSlices {
                    source: TextureSpan {
                        texture: id(IMAGE_A),
                        base_slice: 0,
                        base_level: 0,
                        slice_count: 1,
                        level_count: 0,
                    },
                    dest: TextureSpan {
                        texture: id(IMAGE_B),
                        base_slice: 0,
                        base_level: 0,
                        slice_count: 1,
                        level_count: 0,
                    },
                },
                &residency,
            )
            .err(),
            Some(Refusal::EmptyRange)
        );
    }

    #[test]
    fn a_one_byte_fill_pattern_is_replicated_into_the_word() {
        assert_eq!(fill_word(FillPattern::Byte(0xAB)), 0xABAB_ABAB);
        assert_eq!(fill_word(FillPattern::Pattern4(0x1234_5678)), 0x1234_5678);
    }

    fn arena() -> crate::staging::Arena {
        crate::staging::Arena::adopt(
            256,
            64,
            vec![vk::Buffer::from_raw(0xFF)],
            vec![vk::DeviceMemory::from_raw(0xFF)],
            vec![std::ptr::null_mut()],
        )
    }

    fn span(offset: u64, length: u64) -> BufferSpan {
        BufferSpan {
            buffer: id(BUFFER_A),
            offset,
            length,
        }
    }

    #[test]
    fn a_fill_is_not_a_copy_and_the_copy_planner_says_so() {
        let residency = populated();
        assert_eq!(
            plan(
                &BlitOp::FillBuffer {
                    dest: span(0, 16),
                    pattern: FillPattern::Byte(0),
                },
                &residency,
            )
            .err(),
            Some(Refusal::NotACopy { op: "fill_buffer" })
        );
    }

    #[test]
    fn an_aligned_fill_is_one_command_and_costs_no_scratch() {
        let residency = populated();
        let mut arena = arena();
        let planned = plan_fill(
            span(64, 256),
            FillPattern::Byte(0xFF),
            &residency,
            &mut arena,
        )
        .expect("plannable");
        assert_eq!(planned.dest.as_raw(), 0xB1);
        assert_eq!(planned.head, None);
        assert_eq!(planned.tail, None);
        assert_eq!(
            planned.middle,
            Some(FillRange {
                offset: 64,
                size: 256,
                data: 0xFFFF_FFFF,
            })
        );
        // The common case must not touch the arena at all.
        assert_eq!(arena.census().allocated, 0);
    }

    #[test]
    fn a_ragged_fill_stages_both_edges_and_fills_the_interior() {
        let residency = populated();
        let mut arena = arena();
        // 65..=263: one byte before the boundary at 68, one after the last at
        // 260.
        let planned = plan_fill(
            span(65, 198),
            FillPattern::Byte(0xAB),
            &residency,
            &mut arena,
        )
        .expect("plannable");

        let head = planned.head.expect("a head");
        let tail = planned.tail.expect("a tail");
        let middle = planned.middle.expect("an interior");

        // Exactly the guest's range, once: no byte outside it and none inside
        // it left out.
        assert_eq!(head.dest_offset, 65);
        assert_eq!(head.length, 3);
        assert_eq!(middle.offset, 68);
        assert_eq!(middle.size, 192);
        assert_eq!(tail.dest_offset, 260);
        assert_eq!(tail.length, 3);
        assert_eq!(head.dest_offset + head.length, middle.offset);
        assert_eq!(middle.offset + middle.size, tail.dest_offset);
        assert_eq!(tail.dest_offset + tail.length, 65 + 198);

        assert_eq!(head.byte, 0xAB);
        assert_eq!(tail.byte, 0xAB);
        assert_eq!(middle.data, 0xABAB_ABAB);

        // Six bytes of scratch whatever the size of the fill, in one
        // allocation so there is one failure point.
        assert_eq!(arena.census().allocated, 1);
        assert_eq!(head.window.chunk, tail.window.chunk);
        assert_eq!(head.window.end(), tail.window.offset);
        assert_eq!(head.window.size + tail.window.size, 6);
    }

    #[test]
    fn a_fill_too_short_to_reach_a_boundary_is_all_edge() {
        let residency = populated();
        let mut arena = arena();
        let planned =
            plan_fill(span(1, 2), FillPattern::Byte(7), &residency, &mut arena).expect("plannable");
        let head = planned.head.expect("a head");
        assert_eq!(head.dest_offset, 1);
        assert_eq!(head.length, 2);
        assert_eq!(planned.middle, None);
        assert_eq!(planned.tail, None);
    }

    #[test]
    fn a_ragged_four_byte_pattern_refuses_because_its_phase_is_not_established() {
        let residency = populated();
        let mut arena = arena();
        assert_eq!(
            plan_fill(
                span(1, 16),
                FillPattern::Pattern4(0x1234_5678),
                &residency,
                &mut arena,
            )
            .err(),
            Some(Refusal::RaggedPatternFill {
                offset: 1,
                length: 16,
            })
        );
        // Refused before anything was taken.
        assert_eq!(arena.census().allocated, 0);
        // And the aligned form of the same pattern is fine, which is what
        // makes this a phase question and not a pattern one.
        assert!(plan_fill(
            span(4, 16),
            FillPattern::Pattern4(0x1234_5678),
            &residency,
            &mut arena,
        )
        .is_ok());
    }

    #[test]
    fn a_fill_of_no_bytes_and_one_naming_nothing_refuse_before_the_arena() {
        let residency = populated();
        let mut arena = arena();
        assert_eq!(
            plan_fill(span(0, 0), FillPattern::Byte(0), &residency, &mut arena).err(),
            Some(Refusal::EmptyRange)
        );
        assert_eq!(
            plan_fill(
                BufferSpan {
                    buffer: id(99),
                    offset: 1,
                    length: 8,
                },
                FillPattern::Byte(0),
                &residency,
                &mut arena,
            )
            .err(),
            Some(Refusal::Unresolved {
                miss: Miss::Unknown {
                    slot: ObjectListRef(99)
                }
            })
        );
        assert_eq!(arena.census().allocated, 0);
    }

    #[test]
    fn a_ragged_fill_with_no_scratch_left_refuses_with_the_arenas_own_reason() {
        let residency = populated();
        let mut arena = arena();
        let _ = arena.allocate(256, 1).expect("the whole chunk");
        arena.submitted(TimelinePoint(1));
        let refusal = plan_fill(span(1, 16), FillPattern::Byte(0), &residency, &mut arena)
            .expect_err("nothing left to stage in");
        assert!(matches!(
            refusal,
            Refusal::NoStaging {
                refusal: crate::staging::Refusal::Exhausted { .. }
            }
        ));
        assert!(refusal.to_string().contains("vk_staging_exhausted"));
    }

    #[test]
    fn a_mipmap_generation_is_sent_to_its_own_planner_rather_than_recorded_here() {
        let residency = populated();
        assert_eq!(
            plan(
                &BlitOp::GenerateMipmaps {
                    texture: id(IMAGE_A)
                },
                &residency
            )
            .err(),
            Some(Refusal::NotACopy {
                op: "generate_mipmaps"
            })
        );
        // And it still resolves the name first, so a generation of a resource
        // that does not exist is that refusal and not this one.
        assert!(matches!(
            plan(&BlitOp::GenerateMipmaps { texture: id(99) }, &residency),
            Err(Refusal::Unresolved { .. })
        ));
    }

    #[test]
    fn an_extent_wider_than_a_native_field_refuses_rather_than_truncating() {
        let residency = populated();
        let refusal = plan(
            &BlitOp::TextureRegion {
                source: point(IMAGE_A, 0, 0),
                dest: point(IMAGE_B, 0, 0),
                size: Size3 {
                    width: u64::from(u32::MAX) + 1,
                    height: 1,
                    depth: 1,
                },
                options: Default::default(),
            },
            &residency,
        )
        .expect_err("wider than a u32");
        assert_eq!(
            refusal,
            Refusal::ExtentTooLarge {
                axis: "width",
                value: u64::from(u32::MAX) + 1,
            }
        );
    }

    #[test]
    fn a_region_copy_carries_both_origins_and_both_subresources() {
        let residency = populated();
        let planned = plan(
            &BlitOp::TextureRegion {
                source: TexturePoint {
                    origin: Origin3 { x: 4, y: 8, z: 0 },
                    ..point(IMAGE_A, 2, 1)
                },
                dest: TexturePoint {
                    origin: Origin3 { x: 1, y: 2, z: 0 },
                    ..point(IMAGE_B, 0, 2)
                },
                size: size(8, 4),
                options: Default::default(),
            },
            &residency,
        )
        .expect("plannable");
        let Command::CopyImage {
            source,
            dest,
            regions,
        } = planned
        else {
            panic!("an image copy");
        };
        assert_eq!(source.as_raw(), 0x1A);
        assert_eq!(dest.as_raw(), 0x1B);
        assert_eq!(regions[0].src_offset.x, 4);
        assert_eq!(regions[0].src_offset.y, 8);
        assert_eq!(regions[0].dst_offset.x, 1);
        assert_eq!(regions[0].src_subresource.mip_level, 2);
        assert_eq!(regions[0].src_subresource.base_array_layer, 1);
        assert_eq!(regions[0].dst_subresource.mip_level, 0);
        assert_eq!(regions[0].dst_subresource.base_array_layer, 2);
        assert_eq!(regions[0].extent.width, 8);
        assert_eq!(regions[0].extent.height, 4);
    }

    #[test]
    fn a_stale_resource_lifetime_does_not_survive_into_a_transfer() {
        let mut residency = populated();
        let mut retire = NativeRetirement::new();
        residency
            .delete(id(BUFFER_B), &mut retire)
            .expect("a live name");
        let _ = retire.reached(DeviceEpoch::FIRST, TimelinePoint(1));

        assert!(matches!(
            plan(
                &BlitOp::BufferToBuffer {
                    source: id(BUFFER_A),
                    source_offset: 0,
                    dest: id(BUFFER_B),
                    dest_offset: 0,
                    size: 4,
                },
                &residency,
            ),
            Err(Refusal::Unresolved {
                miss: Miss::Unknown { .. }
            })
        ));
    }

    #[test]
    fn every_refusal_names_itself() {
        let refusals = [
            Refusal::Unresolved {
                miss: Miss::Unknown {
                    slot: ObjectListRef(1),
                },
            },
            Refusal::RowPitchNotWholeBlocks {
                bytes_per_row: 3,
                block_bytes: 4,
            },
            Refusal::ImagePitchNotWholeRows {
                bytes_per_image: 3,
                bytes_per_row: 2,
            },
            Refusal::UnknownFormatGeometry { format: 1 },
            Refusal::RaggedPatternFill {
                offset: 1,
                length: 1,
            },
            Refusal::NoStaging {
                refusal: crate::staging::Refusal::BadAlignment { alignment: 3 },
            },
            Refusal::EmptyRange,
            Refusal::ExtentTooLarge {
                axis: "width",
                value: 1,
            },
            Refusal::NotACopy { op: "x" },
        ];
        let slugs: BTreeSet<&str> = refusals.iter().map(|r| r.slug()).collect();
        assert_eq!(slugs.len(), refusals.len());
        for refusal in refusals {
            assert!(refusal.to_string().starts_with(refusal.slug()));
            assert!(refusal.slug().starts_with("vk_transfer_"));
        }
    }
}
