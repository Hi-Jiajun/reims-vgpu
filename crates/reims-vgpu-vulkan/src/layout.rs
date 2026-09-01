//! Image layouts and queue-family ownership, as explicitly planned transitions.
//!
//! # Why a tracker and not a guess at each use
//!
//! A `VkImage` is in exactly one layout per subresource, and using it in the
//! wrong one is undefined behavior the validation layers catch and the driver
//! sometimes does not. The only way to know the layout is to have tracked every
//! transition, so this tracks them — and the transition it produces names both
//! ends, because a barrier that guessed `oldLayout` would either be wrong or
//! would have to use `VK_IMAGE_LAYOUT_UNDEFINED` and throw the contents away.
//!
//! # Discarding contents is a decision, never an inference
//!
//! A transition out of `UNDEFINED` lets a driver skip decompressing or
//! resolving whatever was there, which is a real saving on a full-surface
//! overwrite. It also destroys the contents. So [`Contents`] is a parameter and
//! not something derived from the use: the caller knows whether the pass loads
//! or clears, and this cannot. An implementation that inferred it would be
//! choosing, silently, to lose bytes the guest wrote.
//!
//! # No transition is the steady state, and it is reported as such
//!
//! [`LayoutTracker::plan`] returns `None` when the subresource is already in
//! the layout the use needs. That is the ordinary case for a draw in a
//! steady-state frame and the reason the tracker earns its keep: the alternative
//! is a barrier per use, every frame, for a layout that never changed.
//!
//! `None` is *not* "no barrier is needed". A read after a write in one layout
//! needs an execution and memory dependency and no layout change, and that
//! dependency is [`crate::barrier`]'s and the dependency compiler's. This module
//! answers layout, and layout only.
//!
//! # Ownership is two halves and both are recorded
//!
//! Moving an image between queue families is a release on the old family and an
//! acquire on the new one, and skipping either makes the contents undefined.
//! [`LayoutTracker::plan_family`] returns both halves together so a caller
//! cannot record one; whether the image is `CONCURRENT` instead is the
//! creation-time decision above this module, and an image declared that way is
//! simply never given a family here.

use ash::vk;
use std::collections::HashMap;

/// A native image this rail owns.
///
/// Minted by the rail rather than derived from a semantic identity: one backing
/// may host several images, and an image may outlive the resource that named it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageId(pub u64);

/// One subresource of one image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Subresource {
    pub level: u32,
    pub layer: u32,
    /// Multi-planar formats lay each plane out separately, so a plane is a
    /// separate subresource and not a coordinate inside one.
    pub plane: u32,
}

impl Subresource {
    #[must_use]
    pub const fn new(level: u32, layer: u32) -> Self {
        Self {
            level,
            layer,
            plane: 0,
        }
    }
}

/// What the image is about to be used for.
///
/// The set is closed on purpose: a use with no entry is a use whose layout this
/// rail has not decided, and adding a variant is where that decision gets made
/// rather than at a call site picking a `VkImageLayout`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Use {
    ColorAttachment,
    DepthStencilAttachment,
    /// Read-only depth or stencil, for a pass that tests without writing.
    DepthStencilRead,
    SampledRead,
    /// A storage image. `GENERAL` because a storage image is written through a
    /// descriptor that has no other optimal layout.
    Storage,
    TransferSrc,
    TransferDst,
    Present,
}

impl Use {
    #[must_use]
    pub const fn layout(self) -> vk::ImageLayout {
        match self {
            Self::ColorAttachment => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            Self::DepthStencilAttachment => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            Self::DepthStencilRead => vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            Self::SampledRead => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            Self::Storage => vk::ImageLayout::GENERAL,
            Self::TransferSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            Self::TransferDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            Self::Present => vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }

    /// The stages that touch the image in this use.
    #[must_use]
    pub const fn stages(self) -> vk::PipelineStageFlags2 {
        match self {
            Self::ColorAttachment => vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            Self::DepthStencilAttachment | Self::DepthStencilRead => {
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            }
            // Any shader stage may sample or store, and which ones is the
            // pipeline's business rather than the layout's. `ALL_COMMANDS` here
            // would be the lazy answer; `ALL_GRAPHICS` plus compute is the
            // honest one for a use that says nothing about the stage.
            Self::SampledRead | Self::Storage => vk::PipelineStageFlags2::from_raw(
                vk::PipelineStageFlags2::ALL_GRAPHICS.as_raw()
                    | vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw(),
            ),
            Self::TransferSrc | Self::TransferDst => vk::PipelineStageFlags2::ALL_TRANSFER,
            // Presentation is not a pipeline stage. The image is handed to the
            // presentation engine, and the dependency is a semaphore rather
            // than a stage mask.
            Self::Present => vk::PipelineStageFlags2::NONE,
        }
    }

    /// The access this use performs.
    #[must_use]
    pub const fn access(self) -> vk::AccessFlags2 {
        match self {
            Self::ColorAttachment => vk::AccessFlags2::from_raw(
                vk::AccessFlags2::COLOR_ATTACHMENT_READ.as_raw()
                    | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw(),
            ),
            Self::DepthStencilAttachment => vk::AccessFlags2::from_raw(
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ.as_raw()
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw(),
            ),
            Self::DepthStencilRead => vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            Self::SampledRead => vk::AccessFlags2::SHADER_SAMPLED_READ,
            Self::Storage => vk::AccessFlags2::from_raw(
                vk::AccessFlags2::SHADER_STORAGE_READ.as_raw()
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw(),
            ),
            Self::TransferSrc => vk::AccessFlags2::TRANSFER_READ,
            Self::TransferDst => vk::AccessFlags2::TRANSFER_WRITE,
            Self::Present => vk::AccessFlags2::NONE,
        }
    }
}

/// Whether the contents survive the transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Contents {
    /// The transition names the current layout, and the driver preserves what
    /// is there.
    Keep,
    /// The transition comes out of `UNDEFINED`. Cheaper, and the previous
    /// contents are gone. Only for a use that overwrites or clears every byte
    /// the caller cares about.
    Discard,
}

/// One planned layout transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "a transition nobody records leaves the image in a layout the next use will not expect"]
pub struct Transition {
    pub image: ImageId,
    pub subresource: Subresource,
    pub from: vk::ImageLayout,
    pub to: vk::ImageLayout,
    pub src_stages: vk::PipelineStageFlags2,
    pub dst_stages: vk::PipelineStageFlags2,
    pub src_access: vk::AccessFlags2,
    pub dst_access: vk::AccessFlags2,
    /// True when `from` is `UNDEFINED` because the caller said the contents
    /// were not needed. Carried so a census can price how much of a frame's
    /// bandwidth the discard path is saving.
    pub discarded_contents: bool,
}

/// Half of a queue-family ownership move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "recording one half of an ownership transfer makes the contents undefined"]
pub struct OwnershipTransfer {
    pub image: ImageId,
    pub subresource: Subresource,
    pub from_family: u32,
    pub to_family: u32,
    /// The layout on both sides. An ownership transfer may change layout too,
    /// but the two halves must agree, so it is one value rather than two.
    pub layout: vk::ImageLayout,
}

/// Why a plan could not be made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decline {
    /// The image was never declared, so its layout is not known and cannot be
    /// assumed. Assuming `UNDEFINED` would discard the contents of an image
    /// somebody else is tracking.
    UnknownImage { image: ImageId },
    /// The subresource is outside what the image was declared with.
    UnknownSubresource {
        image: ImageId,
        subresource: Subresource,
    },
    /// An ownership transfer was asked for on an image with no recorded family.
    /// Either it is `CONCURRENT` and needs none, or its first owner was never
    /// recorded — and guessing which would either emit a transfer that is
    /// invalid usage or skip one that is required.
    NoRecordedFamily { image: ImageId },
}

impl Decline {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::UnknownImage { .. } => "vk_layout_unknown_image",
            Self::UnknownSubresource { .. } => "vk_layout_unknown_subresource",
            Self::NoRecordedFamily { .. } => "vk_layout_no_recorded_family",
        }
    }
}

impl std::fmt::Display for Decline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::UnknownImage { image } | Self::NoRecordedFamily { image } => {
                write!(f, "{} image={}", self.slug(), image.0)
            }
            Self::UnknownSubresource { image, subresource } => write!(
                f,
                "{} image={} level={} layer={} plane={}",
                self.slug(),
                image.0,
                subresource.level,
                subresource.layer,
                subresource.plane
            ),
        }
    }
}

/// What one subresource is doing.
#[derive(Clone, Copy, Debug)]
struct State {
    layout: vk::ImageLayout,
    /// The queue family that owns it, when the image is `EXCLUSIVE`.
    family: Option<u32>,
    /// The stages and access of the last use, which are the source half of the
    /// next transition's dependency.
    last_stages: vk::PipelineStageFlags2,
    last_access: vk::AccessFlags2,
}

#[derive(Clone, Debug)]
struct Image {
    levels: u32,
    layers: u32,
    planes: u32,
    subresources: HashMap<Subresource, State>,
}

/// What the tracker has been asked for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    /// Uses that needed no layout change. The steady state, and the number that
    /// says what the tracker is saving over a barrier per use.
    pub already_in_layout: usize,
    pub transitions: usize,
    /// Transitions the caller declared discardable.
    pub discards: usize,
    pub ownership_transfers: usize,
}

/// Every image's per-subresource layout and family.
#[derive(Debug, Default)]
pub struct LayoutTracker {
    images: HashMap<ImageId, Image>,
    census: Census,
}

impl LayoutTracker {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    /// Declare a freshly created image.
    ///
    /// Every subresource starts `UNDEFINED`, which is what `vkCreateImage`
    /// leaves them in. `family` is the queue family an `EXCLUSIVE` image is
    /// first owned by, or `None` for a `CONCURRENT` one.
    ///
    /// # Panics
    ///
    /// If any dimension is zero. An image with no levels, layers or planes is
    /// not a shallow image; it is one nothing can name a subresource of.
    pub fn declare(
        &mut self,
        image: ImageId,
        levels: u32,
        layers: u32,
        planes: u32,
        family: Option<u32>,
    ) {
        assert!(
            levels > 0 && layers > 0 && planes > 0,
            "an image with no subresources cannot be used"
        );
        let mut subresources = HashMap::new();
        for level in 0..levels {
            for layer in 0..layers {
                for plane in 0..planes {
                    subresources.insert(
                        Subresource {
                            level,
                            layer,
                            plane,
                        },
                        State {
                            layout: vk::ImageLayout::UNDEFINED,
                            family,
                            last_stages: vk::PipelineStageFlags2::NONE,
                            last_access: vk::AccessFlags2::NONE,
                        },
                    );
                }
            }
        }
        self.images.insert(
            image,
            Image {
                levels,
                layers,
                planes,
                subresources,
            },
        );
    }

    /// Forget an image whose native object is gone.
    pub fn forget(&mut self, image: ImageId) {
        self.images.remove(&image);
    }

    /// The layout a subresource is in.
    ///
    /// # Errors
    ///
    /// If the image or the subresource was never declared.
    pub fn layout(
        &self,
        image: ImageId,
        subresource: Subresource,
    ) -> Result<vk::ImageLayout, Decline> {
        Ok(self.state(image, subresource)?.layout)
    }

    /// The queue family that owns a subresource, if the image is `EXCLUSIVE`.
    ///
    /// # Errors
    ///
    /// If the image or the subresource was never declared.
    pub fn family(&self, image: ImageId, subresource: Subresource) -> Result<Option<u32>, Decline> {
        Ok(self.state(image, subresource)?.family)
    }

    /// Plan a use, and record it.
    ///
    /// `Ok(None)` means the subresource is already in the layout the use needs
    /// — the steady state. It does **not** mean no barrier is owed: a hazard in
    /// one layout is still a hazard, and that dependency belongs to
    /// [`crate::barrier`] and the dependency compiler.
    ///
    /// The use is recorded either way, so the next transition's source half is
    /// this use's stages and access rather than a guess.
    ///
    /// # Errors
    ///
    /// If the image or the subresource was never declared.
    pub fn plan(
        &mut self,
        image: ImageId,
        subresource: Subresource,
        use_: Use,
        contents: Contents,
    ) -> Result<Option<Transition>, Decline> {
        let current = *self.state(image, subresource)?;
        let target = use_.layout();
        let discarding = contents == Contents::Discard;
        let transition = if current.layout == target && !discarding {
            self.census.already_in_layout += 1;
            None
        } else {
            self.census.transitions += 1;
            if discarding {
                self.census.discards += 1;
            }
            Some(Transition {
                image,
                subresource,
                from: if discarding {
                    vk::ImageLayout::UNDEFINED
                } else {
                    current.layout
                },
                to: target,
                // Nothing has touched a fresh subresource, so an empty source
                // half is correct there and only there — and a discard is the
                // same statement about the bytes.
                src_stages: if discarding {
                    vk::PipelineStageFlags2::NONE
                } else {
                    current.last_stages
                },
                src_access: if discarding {
                    vk::AccessFlags2::NONE
                } else {
                    current.last_access
                },
                dst_stages: use_.stages(),
                dst_access: use_.access(),
                discarded_contents: discarding,
            })
        };
        let state = self
            .images
            .get_mut(&image)
            .and_then(|i| i.subresources.get_mut(&subresource))
            .expect("just resolved");
        state.layout = target;
        state.last_stages = use_.stages();
        state.last_access = use_.access();
        Ok(transition)
    }

    /// Plan the two halves of a queue-family ownership move, and record it.
    ///
    /// `Ok(None)` when the family already owns it. The two halves come back
    /// together because recording one of them makes the contents undefined.
    ///
    /// # Errors
    ///
    /// If the image or subresource was never declared, or the image has no
    /// recorded family — see [`Decline::NoRecordedFamily`].
    pub fn plan_family(
        &mut self,
        image: ImageId,
        subresource: Subresource,
        to_family: u32,
    ) -> Result<Option<OwnershipTransfer>, Decline> {
        let current = *self.state(image, subresource)?;
        let Some(from_family) = current.family else {
            return Err(Decline::NoRecordedFamily { image });
        };
        if from_family == to_family {
            return Ok(None);
        }
        self.census.ownership_transfers += 1;
        let state = self
            .images
            .get_mut(&image)
            .and_then(|i| i.subresources.get_mut(&subresource))
            .expect("just resolved");
        state.family = Some(to_family);
        Ok(Some(OwnershipTransfer {
            image,
            subresource,
            from_family,
            to_family,
            layout: current.layout,
        }))
    }

    fn state(&self, image: ImageId, subresource: Subresource) -> Result<&State, Decline> {
        let img = self
            .images
            .get(&image)
            .ok_or(Decline::UnknownImage { image })?;
        if subresource.level >= img.levels
            || subresource.layer >= img.layers
            || subresource.plane >= img.planes
        {
            return Err(Decline::UnknownSubresource { image, subresource });
        }
        img.subresources
            .get(&subresource)
            .ok_or(Decline::UnknownSubresource { image, subresource })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMG: ImageId = ImageId(1);

    fn tracker() -> LayoutTracker {
        let mut t = LayoutTracker::new();
        t.declare(IMG, 3, 2, 1, Some(0));
        t
    }

    /// A fresh image is `UNDEFINED`, which is what `vkCreateImage` leaves it in
    /// — not the layout of its first use.
    #[test]
    fn a_fresh_subresource_is_undefined_and_its_first_use_transitions_from_there() {
        let mut t = tracker();
        assert_eq!(
            t.layout(IMG, Subresource::new(0, 0)),
            Ok(vk::ImageLayout::UNDEFINED)
        );
        let transition = t
            .plan(
                IMG,
                Subresource::new(0, 0),
                Use::ColorAttachment,
                Contents::Keep,
            )
            .expect("declared")
            .expect("a transition is needed");
        assert_eq!(transition.from, vk::ImageLayout::UNDEFINED);
        assert_eq!(transition.to, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        assert_eq!(
            transition.src_stages,
            vk::PipelineStageFlags2::NONE,
            "nothing has touched it, so there is nothing to wait for"
        );
        assert!(!transition.discarded_contents, "the caller did not ask to");
    }

    /// The steady state, and the reason the tracker earns its keep.
    #[test]
    fn a_use_in_the_layout_it_is_already_in_needs_no_transition() {
        let mut t = tracker();
        let sub = Subresource::new(0, 0);
        assert!(t
            .plan(IMG, sub, Use::SampledRead, Contents::Keep)
            .expect("declared")
            .is_some_and(|t| t.to == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
        for _ in 0..10 {
            assert_eq!(t.plan(IMG, sub, Use::SampledRead, Contents::Keep), Ok(None));
        }
        assert_eq!(t.census().transitions, 1);
        assert_eq!(t.census().already_in_layout, 10);
    }

    /// The next transition's source half is the last use, not a guess and not
    /// `ALL_COMMANDS`.
    #[test]
    fn a_transition_waits_for_the_use_that_came_before_it() {
        let mut t = tracker();
        let sub = Subresource::new(1, 0);
        t.plan(IMG, sub, Use::ColorAttachment, Contents::Keep)
            .expect("declared");
        let transition = t
            .plan(IMG, sub, Use::SampledRead, Contents::Keep)
            .expect("declared")
            .expect("colour attachment to shader read is a transition");
        assert_eq!(transition.from, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        assert_eq!(transition.to, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(
            transition.src_stages,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
        );
        assert!(transition
            .src_access
            .contains(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE));
        assert!(transition
            .dst_access
            .contains(vk::AccessFlags2::SHADER_SAMPLED_READ));
    }

    /// Discarding is the caller's decision. The saving is real and so is the
    /// loss, which is why nothing here infers it from the use.
    #[test]
    fn a_discard_comes_out_of_undefined_and_waits_for_nothing() {
        let mut t = tracker();
        let sub = Subresource::new(0, 1);
        t.plan(IMG, sub, Use::SampledRead, Contents::Keep)
            .expect("declared");
        let transition = t
            .plan(IMG, sub, Use::ColorAttachment, Contents::Discard)
            .expect("declared")
            .expect("a discard is always a transition");
        assert_eq!(
            transition.from,
            vk::ImageLayout::UNDEFINED,
            "the previous contents are gone"
        );
        assert_eq!(transition.src_stages, vk::PipelineStageFlags2::NONE);
        assert_eq!(transition.src_access, vk::AccessFlags2::NONE);
        assert!(transition.discarded_contents);
        assert_eq!(t.census().discards, 1);
    }

    /// A discard into the layout the image is already in is still a transition:
    /// the point of it is telling the driver the contents are not needed, and
    /// "already in the right layout" would skip exactly that.
    #[test]
    fn a_discard_is_a_transition_even_in_the_same_layout() {
        let mut t = tracker();
        let sub = Subresource::new(2, 1);
        t.plan(IMG, sub, Use::ColorAttachment, Contents::Keep)
            .expect("declared");
        let again = t
            .plan(IMG, sub, Use::ColorAttachment, Contents::Discard)
            .expect("declared")
            .expect("the discard is the whole request");
        assert_eq!(again.from, vk::ImageLayout::UNDEFINED);
        assert_eq!(again.to, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }

    /// Subresources are tracked apart, because they are in separate layouts.
    #[test]
    fn subresources_have_their_own_layouts() {
        let mut t = tracker();
        t.plan(
            IMG,
            Subresource::new(0, 0),
            Use::TransferDst,
            Contents::Discard,
        )
        .expect("declared");
        assert_eq!(
            t.layout(IMG, Subresource::new(0, 0)),
            Ok(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        );
        assert_eq!(
            t.layout(IMG, Subresource::new(1, 0)),
            Ok(vk::ImageLayout::UNDEFINED),
            "a sibling level was not moved"
        );
        assert_eq!(
            t.layout(IMG, Subresource::new(0, 1)),
            Ok(vk::ImageLayout::UNDEFINED),
            "and neither was a sibling layer"
        );
    }

    /// Assuming `UNDEFINED` for an image nobody declared would discard the
    /// contents of an image somebody else is tracking.
    #[test]
    fn an_undeclared_image_or_subresource_is_refused() {
        let mut t = tracker();
        assert_eq!(
            t.plan(
                ImageId(9),
                Subresource::new(0, 0),
                Use::SampledRead,
                Contents::Keep
            ),
            Err(Decline::UnknownImage { image: ImageId(9) })
        );
        let outside = Subresource::new(3, 0);
        assert_eq!(
            t.plan(IMG, outside, Use::SampledRead, Contents::Keep),
            Err(Decline::UnknownSubresource {
                image: IMG,
                subresource: outside
            })
        );
        assert_eq!(
            t.plan(
                IMG,
                Subresource {
                    level: 0,
                    layer: 0,
                    plane: 1
                },
                Use::SampledRead,
                Contents::Keep
            ),
            Err(Decline::UnknownSubresource {
                image: IMG,
                subresource: Subresource {
                    level: 0,
                    layer: 0,
                    plane: 1
                }
            }),
            "a plane the image does not have is a separate layout that does not exist"
        );
    }

    /// Presentation is not a pipeline stage: the image is handed to the
    /// presentation engine and the dependency is a semaphore.
    #[test]
    fn a_present_transition_names_no_stage() {
        let mut t = tracker();
        let sub = Subresource::new(0, 0);
        t.plan(IMG, sub, Use::ColorAttachment, Contents::Discard)
            .expect("declared");
        let transition = t
            .plan(IMG, sub, Use::Present, Contents::Keep)
            .expect("declared")
            .expect("colour attachment to present is a transition");
        assert_eq!(transition.to, vk::ImageLayout::PRESENT_SRC_KHR);
        assert_eq!(transition.dst_stages, vk::PipelineStageFlags2::NONE);
        assert_eq!(transition.dst_access, vk::AccessFlags2::NONE);
        assert!(
            transition
                .src_access
                .contains(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE),
            "and the write it has to wait for is still named"
        );
    }

    /// Both halves together, because recording one makes the contents
    /// undefined.
    #[test]
    fn an_ownership_move_is_one_answer_with_both_families_in_it() {
        let mut t = tracker();
        let sub = Subresource::new(0, 0);
        t.plan(IMG, sub, Use::TransferDst, Contents::Discard)
            .expect("declared");
        assert_eq!(t.family(IMG, sub), Ok(Some(0)));
        let transfer = t
            .plan_family(IMG, sub, 1)
            .expect("declared")
            .expect("family 0 to family 1");
        assert_eq!(transfer.from_family, 0);
        assert_eq!(transfer.to_family, 1);
        assert_eq!(
            transfer.layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            "the two halves must agree on the layout, so it is one value"
        );
        assert_eq!(t.family(IMG, sub), Ok(Some(1)));
        assert_eq!(
            t.plan_family(IMG, sub, 1),
            Ok(None),
            "the family already owns it"
        );
        assert_eq!(t.census().ownership_transfers, 1);
    }

    /// A `CONCURRENT` image needs no transfer, and one whose first owner was
    /// never recorded must not be guessed at either way.
    #[test]
    fn an_image_with_no_recorded_family_refuses_an_ownership_move() {
        let mut t = LayoutTracker::new();
        t.declare(ImageId(2), 1, 1, 1, None);
        let sub = Subresource::new(0, 0);
        assert_eq!(t.family(ImageId(2), sub), Ok(None));
        assert_eq!(
            t.plan_family(ImageId(2), sub, 1),
            Err(Decline::NoRecordedFamily { image: ImageId(2) })
        );
    }

    #[test]
    fn a_forgotten_image_is_no_longer_tracked() {
        let mut t = tracker();
        t.forget(IMG);
        assert_eq!(
            t.layout(IMG, Subresource::new(0, 0)),
            Err(Decline::UnknownImage { image: IMG })
        );
    }

    #[test]
    #[should_panic(expected = "no subresources")]
    fn an_image_with_no_subresources_is_not_a_shallow_image() {
        LayoutTracker::new().declare(IMG, 1, 0, 1, None);
    }

    /// Every use resolves to a layout, and no two uses that need different
    /// treatment share one.
    #[test]
    fn the_uses_that_need_different_layouts_have_different_layouts() {
        let uses = [
            Use::ColorAttachment,
            Use::DepthStencilAttachment,
            Use::DepthStencilRead,
            Use::SampledRead,
            Use::Storage,
            Use::TransferSrc,
            Use::TransferDst,
            Use::Present,
        ];
        let mut layouts: Vec<vk::ImageLayout> = uses.iter().map(|u| u.layout()).collect();
        let count = layouts.len();
        layouts.sort_unstable_by_key(|l| l.as_raw());
        layouts.dedup();
        assert_eq!(layouts.len(), count, "two uses collapsed onto one layout");
        for u in uses {
            assert_ne!(
                u.layout(),
                vk::ImageLayout::UNDEFINED,
                "no use may resolve to the layout that means 'contents gone'"
            );
        }
    }

    #[test]
    fn every_decline_has_its_own_slug() {
        let all = [
            Decline::UnknownImage { image: IMG },
            Decline::UnknownSubresource {
                image: IMG,
                subresource: Subresource::new(0, 0),
            },
            Decline::NoRecordedFamily { image: IMG },
        ];
        let mut slugs: Vec<&str> = all.iter().map(|d| d.slug()).collect();
        slugs.sort_unstable();
        let count = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), count);
    }
}
