//! Where the primitive type is a draw argument on one side and pipeline state
//! on the other, and what that costs.
//!
//! # Metal passes it to the draw; Vulkan builds it into the pipeline
//!
//! `drawPrimitives:` takes an `MTLPrimitiveType`, so a guest may draw
//! triangles and then lines from one render pipeline state without rebuilding
//! anything. `VkGraphicsPipelineCreateInfo` names the topology, so on a bare
//! 1.2 baseline that guest needs a pipeline per type it draws with — the
//! topology is part of the cache key, and a guest that alternates pays a
//! compile the first time it does.
//!
//! `VK_EXT_extended_dynamic_state` — core in 1.3 — makes it
//! `vkCmdSetPrimitiveTopology`, and then one pipeline serves several types.
//! Not all of them: the spec permits changing topology only within one
//! [`TopologyClass`] unless the device also reports
//! `VkPhysicalDeviceExtendedDynamicState3PropertiesEXT::dynamicPrimitiveTopologyUnrestricted`.
//! So the capability has three rungs and they produce three different cache
//! keys, which is what [`key`] returns. Nothing else in this module decides a
//! pipeline's identity, and nothing outside it re-derives one.
//!
//! Getting that wrong is not a validation error on this host — a driver that
//! happens to allow a cross-class change will run it — which is exactly why
//! the rung is asked for structurally rather than assumed.
//!
//! # Metal has no primitive restart
//!
//! There is no restart index in Metal: an index equal to `0xFFFF` or
//! `0xFFFFFFFF` is a vertex like any other. `primitiveRestartEnable` is
//! therefore false for every plan here, including the strips — which are the
//! only topologies it would have affected, and the only ones where leaving it
//! on would silently cut a strip the guest expected continuous.
//!
//! # Planned, not created
//!
//! Nothing here builds a pipeline. Every mapping is tested with no GPU.

use ash::vk;
use reims_vgpu_core::topology::{PrimitiveType, TopologyClass};

/// What this host offers for changing topology without a rebuild.
///
/// Two bits rather than one, because the second is meaningless without the
/// first and the two rungs it makes are different cache keys.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TopologyCell {
    /// `VK_EXT_extended_dynamic_state`'s `extendedDynamicState`, or 1.3 core.
    pub dynamic: bool,
    /// `VkPhysicalDeviceExtendedDynamicState3PropertiesEXT::dynamicPrimitiveTopologyUnrestricted`.
    ///
    /// A *property*, not a feature: a device reports whether it is restricted
    /// rather than being asked to lift the restriction.
    pub unrestricted: bool,
}

/// `MTLPrimitiveType` → `VkPrimitiveTopology`. Total.
///
/// The two enumerations agree numerically across all five, and that is
/// precisely why this is written out: the agreement is a coincidence of two
/// independent specifications rather than a contract, and they already diverge
/// where Metal's set ends — Vulkan continues into `TRIANGLE_FAN`, the four
/// adjacency topologies and `PATCH_LIST`, none of which Metal names. A cast
/// would carry that coincidence forward and turn a future divergence into
/// silently wrong rasterization instead of a compile error.
#[must_use]
pub const fn topology(guest: PrimitiveType) -> vk::PrimitiveTopology {
    match guest {
        PrimitiveType::Point => vk::PrimitiveTopology::POINT_LIST,
        PrimitiveType::Line => vk::PrimitiveTopology::LINE_LIST,
        PrimitiveType::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        PrimitiveType::Triangle => vk::PrimitiveTopology::TRIANGLE_LIST,
        PrimitiveType::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
    }
}

/// What a built pipeline is identified by, as far as topology is concerned.
///
/// The three rungs of [`TopologyCell`], as three different things a cache may
/// key on. Comparing two keys answers "can this pipeline serve that draw"
/// without anybody re-reading the cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TopologyKey {
    /// The baseline: one pipeline per primitive type.
    Exact(PrimitiveType),
    /// Dynamic within a class: one pipeline serves a list and its strip.
    Class(TopologyClass),
    /// Dynamic across classes: one pipeline serves every type.
    Any,
}

/// The key a pipeline drawing `guest` is cached under on this host.
#[must_use]
pub const fn key(guest: PrimitiveType, cell: TopologyCell) -> TopologyKey {
    if !cell.dynamic {
        TopologyKey::Exact(guest)
    } else if cell.unrestricted {
        TopologyKey::Any
    } else {
        TopologyKey::Class(guest.class())
    }
}

/// A pipeline's input-assembly state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputAssemblyPlan {
    pub topology: vk::PrimitiveTopology,
    /// Always false. See the module doc.
    pub primitive_restart_enable: bool,
    /// Whether `vkCmdSetPrimitiveTopology` supplies the topology per draw, in
    /// which case the field above is the pipeline's declared one and the draw
    /// may move within what [`key`] allows.
    pub dynamic: bool,
}

impl InputAssemblyPlan {
    pub const fn native(self) -> vk::PipelineInputAssemblyStateCreateInfo<'static> {
        vk::PipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
            topology: self.topology,
            primitive_restart_enable: vk::FALSE,
            _marker: core::marker::PhantomData,
        }
    }
}

/// Plan the input assembly for a draw of `guest`.
///
/// Total: every primitive type the guest API admits has a plan here. The
/// ordinal that could have failed was closed one layer down.
#[must_use]
pub const fn plan(guest: PrimitiveType, cell: TopologyCell) -> InputAssemblyPlan {
    InputAssemblyPlan {
        topology: topology(guest),
        primitive_restart_enable: false,
        dynamic: cell.dynamic,
    }
}

/// Whether a pipeline built for `built` can draw `wanted` on this host.
///
/// The whole point of [`TopologyKey`]: a caller asks the key, not the cell.
#[must_use]
pub fn serves(built: PrimitiveType, wanted: PrimitiveType, cell: TopologyCell) -> bool {
    key(built, cell) == key(wanted, cell)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    const STATIC: TopologyCell = TopologyCell {
        dynamic: false,
        unrestricted: false,
    };
    const CLASSED: TopologyCell = TopologyCell {
        dynamic: true,
        unrestricted: false,
    };
    const FREE: TopologyCell = TopologyCell {
        dynamic: true,
        unrestricted: true,
    };

    #[test]
    fn every_primitive_type_maps_to_a_distinct_topology() {
        let mapped: BTreeSet<i32> = PrimitiveType::ALL
            .iter()
            .map(|p| topology(*p).as_raw())
            .collect();
        assert_eq!(mapped.len(), PrimitiveType::ALL.len());
        assert_eq!(
            topology(PrimitiveType::LineStrip),
            vk::PrimitiveTopology::LINE_STRIP
        );
        assert_eq!(
            topology(PrimitiveType::TriangleStrip),
            vk::PrimitiveTopology::TRIANGLE_STRIP
        );

        // The five agree numerically today, which is the coincidence the
        // written-out mapping exists to survive: asserted so that a future
        // divergence shows up here as a changed expectation rather than
        // nowhere.
        for guest in PrimitiveType::ALL {
            assert_eq!(topology(guest).as_raw(), guest.ordinal() as i32);
        }
        // And the sets are not the same set: Vulkan continues past Metal's
        // last value, so the agreement is a prefix and nothing more.
        assert_eq!(PrimitiveType::parse(5), None);
        assert_eq!(vk::PrimitiveTopology::TRIANGLE_FAN.as_raw(), 5);
    }

    /// Metal has no restart index, so a strip is never cut. This is the one
    /// place it could have been turned on and the one topology it would have
    /// affected.
    #[test]
    fn no_plan_enables_primitive_restart_including_the_strips() {
        for guest in PrimitiveType::ALL {
            for cell in [STATIC, CLASSED, FREE] {
                let plan = plan(guest, cell);
                assert!(!plan.primitive_restart_enable);
                assert_eq!(plan.native().primitive_restart_enable, vk::FALSE);
            }
        }
        // Non-vacuous: two of those five were strips.
        assert_eq!(
            PrimitiveType::ALL.iter().filter(|p| p.is_strip()).count(),
            2
        );
    }

    /// Without dynamic topology every type is its own pipeline; that is the
    /// baseline cost, asserted rather than assumed.
    #[test]
    fn the_baseline_needs_a_pipeline_per_primitive_type() {
        let keys: BTreeSet<TopologyKey> =
            PrimitiveType::ALL.iter().map(|p| key(*p, STATIC)).collect();
        assert_eq!(keys.len(), 5);
        for a in PrimitiveType::ALL {
            for b in PrimitiveType::ALL {
                assert_eq!(serves(a, b, STATIC), a == b);
            }
        }
    }

    /// With `extendedDynamicState` and nothing more, a pipeline serves its
    /// class and no further. Five types collapse to three pipelines.
    #[test]
    fn dynamic_topology_serves_a_class_and_stops_at_its_edge() {
        let keys: BTreeSet<TopologyKey> = PrimitiveType::ALL
            .iter()
            .map(|p| key(*p, CLASSED))
            .collect();
        assert_eq!(keys.len(), 3);
        assert!(serves(
            PrimitiveType::Line,
            PrimitiveType::LineStrip,
            CLASSED
        ));
        assert!(serves(
            PrimitiveType::Triangle,
            PrimitiveType::TriangleStrip,
            CLASSED
        ));
        // The edge. A driver that happens to allow this would run it, which is
        // why the restriction is asked for rather than discovered.
        assert!(!serves(
            PrimitiveType::Triangle,
            PrimitiveType::Line,
            CLASSED
        ));
        assert!(!serves(PrimitiveType::Point, PrimitiveType::Line, CLASSED));
        for guest in PrimitiveType::ALL {
            assert!(plan(guest, CLASSED).dynamic);
        }
    }

    /// With the unrestricted property one pipeline serves everything.
    #[test]
    fn an_unrestricted_host_needs_one_pipeline() {
        let keys: BTreeSet<TopologyKey> =
            PrimitiveType::ALL.iter().map(|p| key(*p, FREE)).collect();
        assert_eq!(keys, BTreeSet::from([TopologyKey::Any]));
        for a in PrimitiveType::ALL {
            for b in PrimitiveType::ALL {
                assert!(serves(a, b, FREE));
            }
        }
    }

    /// The property means nothing without the feature: a device reporting
    /// `unrestricted` while `extendedDynamicState` is off is still on the
    /// baseline rung, because there is no dynamic state for it to unrestrict.
    #[test]
    fn the_property_alone_does_not_lift_anything() {
        let confused = TopologyCell {
            dynamic: false,
            unrestricted: true,
        };
        assert_eq!(
            key(PrimitiveType::Line, confused),
            TopologyKey::Exact(PrimitiveType::Line)
        );
        assert!(!serves(
            PrimitiveType::Line,
            PrimitiveType::LineStrip,
            confused
        ));
        assert!(!plan(PrimitiveType::Line, confused).dynamic);
    }

    #[test]
    fn the_native_state_carries_the_plan() {
        let native = plan(PrimitiveType::TriangleStrip, CLASSED).native();
        assert_eq!(native.topology, vk::PrimitiveTopology::TRIANGLE_STRIP);
        assert_eq!(native.primitive_restart_enable, vk::FALSE);
        assert_eq!(
            native.s_type,
            vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO
        );
    }
}
