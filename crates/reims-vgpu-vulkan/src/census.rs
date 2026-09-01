//! What this physical device offers, taken once, and the floor it has to clear
//! to be used at all.
//!
//! # One snapshot, because two queries of one fact can disagree
//!
//! Every capability decision this rail makes — which memory class an allocation
//! gets, which queue it submits to, which mechanism carries a descriptor,
//! whether a barrier can name the mesh stage — reads from a [`Census`] taken
//! once at device bring-up and never re-queried. A rail that asked the driver
//! again at each decision would be one where a lost device, a re-enumerated
//! extension list or a differently-populated feature chain could make the same
//! guest stream mean two things in one run.
//!
//! [`Census`] is therefore immutable, `Copy`, and has no method that asks the
//! driver anything. It is the *catalog* the architecture allows to be
//! process-global: no command pools, no queues, no resources, no submission
//! state, no mutable policy.
//!
//! # A name is not a capability, and this type cannot hold one
//!
//! The rule is that gates are on structural capabilities and never on a driver
//! name, a vendor id, an API implementation, or `VK_KHR_portability_subset`.
//! That rule is enforced here by absence: [`Reported`] has no field for a
//! vendor id, a device id, a device name, or a driver id, so a gate downstream
//! of this module has nothing to branch on even if somebody wanted to. What
//! comes out is a set of measured facts, and the census is the only thing the
//! rest of the crate is given.
//!
//! # An extension in the list is not a feature
//!
//! `vkEnumerateDeviceExtensionProperties` says the entry points can be
//! resolved. It does not say the feature bit is set, and a rail that used a
//! feature it did not enable is one whose failures are validation errors on
//! somebody else's machine. So every optional capability here needs both: the
//! extension present *and* the feature reported. [`Census::take`] is where the
//! two are joined, and it is the only place.
//!
//! # The floor is queried, not implied
//!
//! Timeline semaphores are a support floor for this rail, and Vulkan 1.2 being
//! the baseline does not supply them: `timelineSemaphore` is a feature that
//! must be queried and enabled. So a device that reports 1.2 with the feature
//! off is refused with [`Floor::NoTimelineSemaphores`] rather than quietly
//! taking a blocking drain path — the architecture is explicit that no such
//! compatibility path is installed, because installing one makes the fast path
//! untested on exactly the hosts that need it.

use crate::barrier::StageSupport;
use crate::descriptor::DescriptorCell;
use crate::memory::{classify_memory, MemoryProfile};
use crate::placement::HostCell;
use crate::queues::{self, QueueChoice};
use ash::vk;

/// A Vulkan API version, as the two numbers a floor is expressed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ApiVersion {
    pub major: u32,
    pub minor: u32,
}

impl ApiVersion {
    /// The version the device reports in `VkPhysicalDeviceProperties::apiVersion`.
    ///
    /// The patch and variant fields are dropped deliberately: no decision this
    /// rail makes is allowed to turn on a patch level, and keeping them would
    /// make `ApiVersion`'s ordering compare things no gate may consult.
    #[must_use]
    pub const fn decode(packed: u32) -> Self {
        Self {
            major: vk::api_version_major(packed),
            minor: vk::api_version_minor(packed),
        }
    }

    #[must_use]
    pub const fn at_least(self, major: u32, minor: u32) -> bool {
        self.major > major || (self.major == major && self.minor >= minor)
    }
}

impl std::fmt::Display for ApiVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// Extension names this census asks about.
///
/// Spelled once here rather than at each comparison, so a typo is a compile
/// error in one place instead of a capability that is silently always absent.
///
/// `VK_KHR_timeline_semaphore` is deliberately absent: 1.2 is the baseline and
/// promoted it, so the extension name is never the question — the feature bit
/// is, and that is a field on [`Reported`].
pub mod extension {
    pub const PUSH_DESCRIPTOR: &str = "VK_KHR_push_descriptor";
    pub const DESCRIPTOR_BUFFER: &str = "VK_EXT_descriptor_buffer";
    pub const MESH_SHADER: &str = "VK_EXT_mesh_shader";
    pub const EXTERNAL_MEMORY_HOST: &str = "VK_EXT_external_memory_host";
    pub const SYNCHRONIZATION_2: &str = "VK_KHR_synchronization2";
    pub const SWAPCHAIN: &str = "VK_KHR_swapchain";
}

/// What the driver said, as the caller read it off the device.
///
/// A borrowed view rather than an owned struct: the caller already holds the
/// enumerated extension list and the property structures, and copying them
/// would create a second place a fact lives.
///
/// Deliberately has no vendor id, device id, device name or driver id. See the
/// module doc.
#[derive(Clone, Copy, Debug)]
pub struct Reported<'a> {
    /// `VkPhysicalDeviceProperties::apiVersion`, packed.
    pub api_version: u32,
    /// Device extension names, exactly as enumerated.
    pub extensions: &'a [&'a str],
    /// `VkPhysicalDeviceVulkan12Features::timelineSemaphore`.
    pub timeline_semaphore: bool,
    /// `VkPhysicalDeviceSynchronization2Features::synchronization2`.
    pub synchronization2: bool,
    /// `VkPhysicalDeviceMeshShaderFeaturesEXT::meshShader`.
    pub mesh_shader: bool,
    /// `VkPhysicalDeviceDescriptorBufferFeaturesEXT::descriptorBuffer`.
    pub descriptor_buffer: bool,
    /// `VkPhysicalDevicePushDescriptorPropertiesKHR::maxPushDescriptors`.
    pub max_push_descriptors: u32,
    pub memory: &'a vk::PhysicalDeviceMemoryProperties,
    pub queue_families: &'a [vk::QueueFamilyProperties],
}

impl Reported<'_> {
    /// Whether the device enumerated an extension.
    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        self.extensions.contains(&name)
    }
}

/// Why this device cannot be used at all.
///
/// A floor failure is a typed refusal and never a degraded mode. Each variant
/// carries what was reported, so the refusal says which fact was missing rather
/// than that a device "was not suitable".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Floor {
    /// Vulkan 1.2 is the baseline on every supported host.
    ApiTooOld { reported: ApiVersion },
    /// The device reports a new enough API but has `timelineSemaphore` off.
    ///
    /// Distinct from [`Self::ApiTooOld`] precisely because it is the case the
    /// version number does not cover; see the module doc.
    NoTimelineSemaphores { reported: ApiVersion },
    /// No queue family can both draw and dispatch.
    NoUsableQueue { decline: queues::Decline },
    /// The device cannot present.
    NoSwapchain,
}

impl Floor {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::ApiTooOld { .. } => "vk_census_api_too_old",
            Self::NoTimelineSemaphores { .. } => "vk_census_no_timeline_semaphores",
            Self::NoUsableQueue { .. } => "vk_census_no_usable_queue",
            Self::NoSwapchain => "vk_census_no_swapchain",
        }
    }
}

impl std::fmt::Display for Floor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiTooOld { reported } | Self::NoTimelineSemaphores { reported } => {
                write!(f, "{} reported={reported}", self.slug())
            }
            Self::NoUsableQueue { decline } => write!(f, "{} {decline}", self.slug()),
            Self::NoSwapchain => f.write_str(self.slug()),
        }
    }
}

/// Whether descriptor buffers have been qualified on this host.
///
/// The architecture asks for descriptor buffers where the driver reports
/// support *and* validation and performance tests pass. The second half is a
/// measurement, not something a driver reports, so it cannot come out of
/// [`Census::take`] — and it is a separate type rather than a `bool` so that
/// "nobody has measured this" is a state, distinct from "measured and failed".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DescriptorBufferProbe {
    /// No qualification has run here. The conservative state, and the one every
    /// freshly taken census is in.
    #[default]
    NotRun,
    Passed,
    Failed,
}

/// The measured facts about one physical device.
///
/// Immutable and `Copy`. Every accessor is a projection of what was taken; none
/// asks the driver anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Census {
    api: ApiVersion,
    memory: MemoryProfile,
    queues: QueueChoice,
    stages: StageSupport,
    descriptors: DescriptorCell,
    host_pointer_import: bool,
    synchronization2: bool,
    can_present: bool,
}

impl Census {
    /// Take the snapshot, or refuse the device.
    ///
    /// The one place an extension name and a feature bit are joined, and the
    /// one place the support floor is checked.
    ///
    /// # Errors
    ///
    /// [`Floor`] when the device cannot serve this rail at all.
    pub fn take(reported: Reported<'_>) -> Result<Self, Floor> {
        let api = ApiVersion::decode(reported.api_version);
        if !api.at_least(1, 2) {
            return Err(Floor::ApiTooOld { reported: api });
        }
        // The floor, queried. 1.2 promoted the extension, so a 1.2 device need
        // not enumerate `VK_KHR_timeline_semaphore` — but it must still report
        // the feature, and that is the fact checked.
        if !reported.timeline_semaphore {
            return Err(Floor::NoTimelineSemaphores { reported: api });
        }
        if !reported.has(extension::SWAPCHAIN) {
            return Err(Floor::NoSwapchain);
        }

        let families = queues::families(reported.queue_families);
        let queues = QueueChoice::from_families(&families)
            .map_err(|decline| Floor::NoUsableQueue { decline })?;

        // Both halves, every time: the entry points resolve *and* the feature
        // is on. Either alone is a capability this rail must not use.
        let mesh_shader = reported.has(extension::MESH_SHADER) && reported.mesh_shader;
        let descriptor_buffer =
            reported.has(extension::DESCRIPTOR_BUFFER) && reported.descriptor_buffer;
        // Push descriptors have no feature bit of their own in the extension;
        // the property is the capability, and a limit of zero is checked where
        // the tier is chosen rather than here, so the refusal can say which of
        // the two facts was missing.
        let push_descriptor = reported.has(extension::PUSH_DESCRIPTOR) || api.at_least(1, 4);

        Ok(Self {
            api,
            memory: classify_memory(reported.memory),
            queues,
            stages: StageSupport { mesh_shader },
            descriptors: DescriptorCell {
                push_descriptor,
                max_push_descriptors: reported.max_push_descriptors,
                descriptor_buffer,
                // Never reported by a driver. See [`DescriptorBufferProbe`].
                descriptor_buffer_qualified: false,
            },
            host_pointer_import: reported.has(extension::EXTERNAL_MEMORY_HOST),
            // 1.3 promoted it to core, so a 1.3 device has it whether or not it
            // enumerates the extension. Below that both facts are needed, for
            // the reason above.
            synchronization2: api.at_least(1, 3)
                || (reported.has(extension::SYNCHRONIZATION_2) && reported.synchronization2),
            can_present: true,
        })
    }

    /// Record a descriptor-buffer qualification result.
    ///
    /// Consuming and returning `Self` rather than mutating: a census is one
    /// snapshot, and a probe result taken later produces a new one rather than
    /// changing what an earlier decision was made against.
    ///
    /// Only [`DescriptorBufferProbe::Passed`] can raise the tier, and it cannot
    /// raise it past what the device reported — a probe that passed on a device
    /// without the extension leaves the cell exactly as it was, because
    /// [`Self::take`] already decided that fact.
    #[must_use]
    pub const fn with_descriptor_buffer_probe(mut self, probe: DescriptorBufferProbe) -> Self {
        self.descriptors.descriptor_buffer_qualified =
            self.descriptors.descriptor_buffer && matches!(probe, DescriptorBufferProbe::Passed);
        self
    }

    #[must_use]
    pub const fn api(&self) -> ApiVersion {
        self.api
    }

    #[must_use]
    pub const fn memory(&self) -> MemoryProfile {
        self.memory
    }

    /// The immutable half of the queue decision. A device epoch turns it into
    /// a [`crate::queues::QueuePlan`] with [`QueuePlan::adopt`]; the catalog
    /// holds no ownership state.
    ///
    /// [`QueuePlan::adopt`]: crate::queues::QueuePlan::adopt
    #[must_use]
    pub const fn queues(&self) -> QueueChoice {
        self.queues
    }

    #[must_use]
    pub const fn stages(&self) -> StageSupport {
        self.stages
    }

    #[must_use]
    pub const fn descriptors(&self) -> DescriptorCell {
        self.descriptors
    }

    /// The cell [`crate::placement`] decides a route from.
    #[must_use]
    pub const fn host_cell(&self) -> HostCell {
        HostCell {
            topology: self.memory.topology,
            host_pointer_import: self.host_pointer_import,
        }
    }

    /// Whether barriers may use the `synchronization2` commands, or have to go
    /// through [`crate::barrier::BarrierPlan::legacy`].
    #[must_use]
    pub const fn synchronization2(&self) -> bool {
        self.synchronization2
    }

    #[must_use]
    pub const fn can_present(&self) -> bool {
        self.can_present
    }

    /// One line naming every fact a decision here is allowed to read.
    ///
    /// The architecture asks for the exact capability census to be recorded
    /// with every run, and this is it. It names no device, because there is no
    /// device name in the census to name.
    #[must_use]
    pub fn report_line(&self) -> String {
        format!(
            "vk_census api={} topology={} signal={} import={} sync2={} mesh={} push={} \
             push_max={} desc_buffer={} desc_qualified={} queue_family={} compute={}",
            self.api,
            self.memory.topology.slug(),
            self.memory.signal.slug(),
            self.host_pointer_import,
            self.synchronization2,
            self.stages.mesh_shader,
            self.descriptors.push_descriptor,
            self.descriptors.max_push_descriptors,
            self.descriptors.descriptor_buffer,
            self.descriptors.descriptor_buffer_qualified,
            self.queues.universal().index,
            self.queues.compute(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{select, Narrowing, Tier};
    use crate::memory::{fixtures as mem, MemoryTopology};

    /// Queue families as the driver reports them, rather than as
    /// [`crate::queues::Family`] — the census takes what the driver said.
    fn family(flags: vk::QueueFlags, count: u32) -> vk::QueueFamilyProperties {
        vk::QueueFamilyProperties {
            queue_flags: flags,
            queue_count: count,
            timestamp_valid_bits: 64,
            min_image_transfer_granularity: vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
        }
    }

    /// One universal family plus a dedicated copy engine.
    fn discrete_families() -> Vec<vk::QueueFamilyProperties> {
        vec![
            family(
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
                16,
            ),
            family(vk::QueueFlags::TRANSFER, 2),
        ]
    }

    /// One universal family, which is most integrated parts.
    fn integrated_families() -> Vec<vk::QueueFamilyProperties> {
        vec![family(
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
            1,
        )]
    }

    fn packed(major: u32, minor: u32) -> u32 {
        vk::make_api_version(0, major, minor, 0)
    }

    const BASELINE: &[&str] = &[extension::SWAPCHAIN];

    fn reported<'a>(
        api: u32,
        extensions: &'a [&'a str],
        memory: &'a vk::PhysicalDeviceMemoryProperties,
        families: &'a [vk::QueueFamilyProperties],
    ) -> Reported<'a> {
        Reported {
            api_version: api,
            extensions,
            timeline_semaphore: true,
            synchronization2: false,
            mesh_shader: false,
            descriptor_buffer: false,
            max_push_descriptors: 0,
            memory,
            queue_families: families,
        }
    }

    #[test]
    fn a_baseline_device_is_admitted_and_takes_the_bottom_rung_of_everything() {
        let memory = mem::nvidia_discrete();
        let families = discrete_families();
        let census = Census::take(reported(packed(1, 2), BASELINE, &memory, &families))
            .expect("1.2 with timeline semaphores is the baseline");

        assert_eq!(census.api(), ApiVersion { major: 1, minor: 2 });
        assert_eq!(census.memory().topology, MemoryTopology::Discrete);
        assert!(!census.synchronization2(), "1.2 has no core sync2");
        assert!(!census.stages().mesh_shader);
        assert!(!census.host_cell().host_pointer_import);
        assert_eq!(
            select(census.descriptors(), Narrowing::default()).tier,
            Tier::PooledSets
        );
    }

    #[test]
    fn a_device_below_the_baseline_is_refused_by_version() {
        let memory = mem::nvidia_discrete();
        let families = discrete_families();
        let floor = Census::take(reported(packed(1, 1), BASELINE, &memory, &families))
            .expect_err("1.1 is below the baseline");
        assert_eq!(
            floor,
            Floor::ApiTooOld {
                reported: ApiVersion { major: 1, minor: 1 }
            }
        );
        assert_eq!(floor.slug(), "vk_census_api_too_old");
    }

    /// The floor the version number does not cover.
    #[test]
    fn a_new_enough_device_with_the_feature_off_is_still_refused() {
        let memory = mem::nvidia_discrete();
        let families = discrete_families();
        let mut r = reported(packed(1, 3), BASELINE, &memory, &families);
        r.timeline_semaphore = false;
        let floor = Census::take(r).expect_err("the feature is the floor, not the version");
        assert_eq!(
            floor,
            Floor::NoTimelineSemaphores {
                reported: ApiVersion { major: 1, minor: 3 }
            }
        );
        assert!(floor.to_string().contains("1.3"));
    }

    #[test]
    fn a_device_that_cannot_present_is_refused_before_anything_is_measured() {
        let memory = mem::nvidia_discrete();
        let families = discrete_families();
        let floor = Census::take(reported(packed(1, 2), &[], &memory, &families))
            .expect_err("no swapchain");
        assert_eq!(floor, Floor::NoSwapchain);
    }

    #[test]
    fn a_device_with_no_drawing_queue_is_refused_and_says_which_decline() {
        let memory = mem::nvidia_discrete();
        let families: Vec<vk::QueueFamilyProperties> = Vec::new();
        let floor = Census::take(reported(packed(1, 2), BASELINE, &memory, &families))
            .expect_err("no queue family at all");
        assert!(matches!(floor, Floor::NoUsableQueue { .. }));
        assert_eq!(floor.slug(), "vk_census_no_usable_queue");
    }

    /// The claim the module exists to make structural: the entry points
    /// resolving is not the feature being on.
    #[test]
    fn an_extension_without_its_feature_is_not_a_capability() {
        let memory = mem::apple_m3_max();
        let families = integrated_families();
        let extensions = &[
            extension::SWAPCHAIN,
            extension::MESH_SHADER,
            extension::DESCRIPTOR_BUFFER,
        ];

        let mut r = reported(packed(1, 2), extensions, &memory, &families);
        r.mesh_shader = false;
        r.descriptor_buffer = false;
        let without = Census::take(r).expect("admitted");
        assert!(!without.stages().mesh_shader);
        assert!(!without.descriptors().descriptor_buffer);

        r.mesh_shader = true;
        r.descriptor_buffer = true;
        let with = Census::take(r).expect("admitted");
        assert!(with.stages().mesh_shader);
        assert!(with.descriptors().descriptor_buffer);
    }

    /// And the mirror: a feature bit set with no extension enumerated is not a
    /// capability either.
    #[test]
    fn a_feature_without_its_extension_is_not_a_capability() {
        let memory = mem::apple_m3_max();
        let families = integrated_families();
        let mut r = reported(packed(1, 2), BASELINE, &memory, &families);
        r.mesh_shader = true;
        r.descriptor_buffer = true;
        r.synchronization2 = true;
        let census = Census::take(r).expect("admitted");
        assert!(!census.stages().mesh_shader);
        assert!(!census.descriptors().descriptor_buffer);
        assert!(!census.synchronization2());
    }

    #[test]
    fn a_promoted_capability_needs_no_extension_at_its_core_version() {
        let memory = mem::apple_m3_max();
        let families = integrated_families();
        // 1.3 has synchronization2 in core; the extension is not enumerated and
        // the feature struct is not chained.
        let census =
            Census::take(reported(packed(1, 3), BASELINE, &memory, &families)).expect("admitted");
        assert!(census.synchronization2());
        // And below it, the extension route reaches the same answer.
        let mut r = reported(
            packed(1, 2),
            &[extension::SWAPCHAIN, extension::SYNCHRONIZATION_2],
            &memory,
            &families,
        );
        r.synchronization2 = true;
        assert!(Census::take(r).expect("admitted").synchronization2());
    }

    /// The M-series shape, measured rather than named: unified memory, host
    /// pointer import, push descriptors, no descriptor buffers.
    #[test]
    fn a_unified_import_capable_host_reaches_the_push_rung_and_the_direct_route() {
        let memory = mem::apple_m3_max();
        let families = integrated_families();
        let mut r = reported(
            packed(1, 2),
            &[
                extension::SWAPCHAIN,
                extension::PUSH_DESCRIPTOR,
                extension::EXTERNAL_MEMORY_HOST,
            ],
            &memory,
            &families,
        );
        r.max_push_descriptors = 32;
        let census = Census::take(r).expect("admitted");

        assert_eq!(census.memory().topology, MemoryTopology::Unified);
        assert!(census.host_cell().host_pointer_import);
        assert_eq!(
            select(census.descriptors(), Narrowing::default()).tier,
            Tier::PushDescriptor { max: 32 }
        );
    }

    #[test]
    fn a_probe_cannot_qualify_what_the_device_does_not_report() {
        let memory = mem::intel_igpu();
        let families = integrated_families();
        let census = Census::take(reported(packed(1, 2), BASELINE, &memory, &families))
            .expect("admitted")
            .with_descriptor_buffer_probe(DescriptorBufferProbe::Passed);
        assert!(
            !census.descriptors().descriptor_buffer_qualified,
            "a probe result cannot conjure the extension"
        );
        assert_ne!(
            select(census.descriptors(), Narrowing::default()).tier,
            Tier::DescriptorBuffer
        );
    }

    #[test]
    fn a_probe_is_what_raises_a_reported_device_to_the_top_rung() {
        let memory = mem::intel_igpu();
        let families = integrated_families();
        let mut r = reported(
            packed(1, 3),
            &[extension::SWAPCHAIN, extension::DESCRIPTOR_BUFFER],
            &memory,
            &families,
        );
        r.descriptor_buffer = true;
        let census = Census::take(r).expect("admitted");

        // Reported, unmeasured: not the top rung.
        assert_eq!(
            select(census.descriptors(), Narrowing::default()).tier,
            Tier::PooledSets
        );
        for probe in [DescriptorBufferProbe::NotRun, DescriptorBufferProbe::Failed] {
            let tried = census.with_descriptor_buffer_probe(probe);
            assert!(
                !tried.descriptors().descriptor_buffer_qualified,
                "{probe:?}"
            );
        }
        let qualified = census.with_descriptor_buffer_probe(DescriptorBufferProbe::Passed);
        assert_eq!(
            select(qualified.descriptors(), Narrowing::default()).tier,
            Tier::DescriptorBuffer
        );
    }

    #[test]
    fn the_report_line_names_every_fact_and_no_device() {
        let memory = mem::apple_m3_max();
        let families = integrated_families();
        let mut r = reported(
            packed(1, 2),
            &[
                extension::SWAPCHAIN,
                extension::PUSH_DESCRIPTOR,
                extension::EXTERNAL_MEMORY_HOST,
            ],
            &memory,
            &families,
        );
        r.max_push_descriptors = 32;
        let line = Census::take(r).expect("admitted").report_line();

        for fact in [
            "api=1.2",
            "topology=unified",
            "import=true",
            "sync2=false",
            "mesh=false",
            "push=true",
            "push_max=32",
            "desc_buffer=false",
            "desc_qualified=false",
        ] {
            assert!(line.contains(fact), "{fact} missing from {line}");
        }
    }

    #[test]
    fn a_version_orders_by_major_then_minor_and_ignores_the_patch() {
        let a = ApiVersion::decode(vk::make_api_version(0, 1, 2, 198));
        let b = ApiVersion::decode(vk::make_api_version(0, 1, 2, 0));
        assert_eq!(a, b, "no gate may turn on a patch level");
        assert!(a.at_least(1, 2));
        assert!(!a.at_least(1, 3));
        assert!(a.at_least(1, 0));
        assert!(ApiVersion::decode(vk::make_api_version(0, 2, 0, 0)).at_least(1, 9));
    }
}
