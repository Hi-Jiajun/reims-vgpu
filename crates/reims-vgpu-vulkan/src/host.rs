//! The instance, the physical device this rail bound, and the capability
//! catalog taken off it.
//!
//! # What may live here, and what may not
//!
//! The architecture allows exactly one process-global Vulkan thing: immutable
//! host discovery. [`VulkanHost`] is it — the initialized `VkInstance`, the
//! loaded function tables, the selected `VkPhysicalDevice` identity, and the
//! [`Census`]. It has no command pools, no queues, no resources, no caches, no
//! submission state, and no mutable policy, so a device epoch that is lost or
//! recreated cannot couple two sessions through it. Every mutable thing in this
//! crate is owned per epoch or per worker instead.
//!
//! # Selection ranks classes, and the class is not a name
//!
//! Enumerating physical devices on a hybrid laptop returns two GPUs and on a
//! Mesa host it can return a software rasterizer ahead of a real one. Choosing
//! the first is how a run silently binds llvmpipe and reports a frame rate
//! nobody can explain.
//!
//! So selection has two steps, and only the first can exclude anything:
//!
//! 1. **Admission** is [`Census::take`], and it is entirely capability-driven.
//!    A device that fails it is refused with the fact it was missing.
//! 2. **Preference** among the admitted is [`DeviceClass`], which is
//!    `VkPhysicalDeviceType` — a *reported class*, not a marketing name, a
//!    vendor id or a driver id. It decides only the order among devices that
//!    are all already usable, and it can never make one usable or unusable.
//!    Ties keep the first enumerated device, so the order is total and a rerun
//!    on one host picks the same device.
//!
//! That split is why the census carries no device identity: the one place a
//! reported class is consulted is here, for ordering, and nothing downstream is
//! given it.
//!
//! # A feature struct is chained only when its extension is there
//!
//! `vkGetPhysicalDeviceFeatures2` is asked about an optional feature only after
//! the extension has been found in the enumerated list. Chaining a struct for
//! an extension the device does not implement asks a driver about something it
//! has never heard of, and the answer to that is at best a zero and at worst a
//! validation error on somebody else's machine. It is also the same discipline
//! [`Census::take`] applies from the other side.

use crate::census::{extension, ApiVersion, Census, Floor, Reported};
use ash::vk;
use std::ffi::{CStr, CString};

/// The class of device the driver reports itself as.
///
/// `VkPhysicalDeviceType` and nothing else. It orders devices that have all
/// already been admitted on capability; it never admits or refuses one. See the
/// module doc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceClass {
    Discrete,
    Integrated,
    Virtual,
    Cpu,
    Other,
}

impl DeviceClass {
    #[must_use]
    pub fn of(reported: vk::PhysicalDeviceType) -> Self {
        match reported {
            vk::PhysicalDeviceType::DISCRETE_GPU => Self::Discrete,
            vk::PhysicalDeviceType::INTEGRATED_GPU => Self::Integrated,
            vk::PhysicalDeviceType::VIRTUAL_GPU => Self::Virtual,
            vk::PhysicalDeviceType::CPU => Self::Cpu,
            _ => Self::Other,
        }
    }

    /// Preference, higher first.
    ///
    /// A CPU device ranks last rather than being excluded: a software
    /// rasterizer that clears the floor is a legal host for this rail and the
    /// only host on some machines, and refusing it would turn "slow" into
    /// "nothing at all". What it must not do is win against real hardware that
    /// enumerated after it.
    #[must_use]
    pub const fn rank(self) -> u8 {
        match self {
            Self::Discrete => 4,
            Self::Integrated => 3,
            Self::Virtual => 2,
            Self::Other => 1,
            Self::Cpu => 0,
        }
    }

    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Discrete => "discrete",
            Self::Integrated => "integrated",
            Self::Virtual => "virtual",
            Self::Cpu => "cpu",
            Self::Other => "other",
        }
    }
}

/// One enumerated device, judged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// Position in the enumeration, which is what breaks a rank tie.
    pub index: usize,
    pub class: DeviceClass,
    /// What the census made of it: the catalog, or the floor it failed.
    pub verdict: Result<Census, Floor>,
}

/// No enumerated device can serve this rail.
///
/// Carries every candidate's reason rather than a count, because the interesting
/// case is a host with two GPUs where each lost for a different fact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoDevice {
    pub enumerated: usize,
    pub floors: Vec<Floor>,
}

impl NoDevice {
    #[must_use]
    pub const fn slug(&self) -> &'static str {
        "vk_host_no_usable_device"
    }
}

impl std::fmt::Display for NoDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} enumerated={}", self.slug(), self.enumerated)?;
        for (index, floor) in self.floors.iter().enumerate() {
            write!(f, " [{index}]={floor}")?;
        }
        Ok(())
    }
}

/// Pick the device this rail binds.
///
/// Pure over the judged candidates, so every hybrid, software-rasterizer and
/// below-floor arrangement is testable with no device present.
///
/// # Errors
///
/// [`NoDevice`] when nothing cleared the floor, naming what each candidate was
/// missing.
pub fn select(candidates: &[Candidate]) -> Result<Candidate, NoDevice> {
    candidates
        .iter()
        .filter(|c| c.verdict.is_ok())
        // `max_by_key` keeps the *last* maximum, so the key carries the
        // negated index to make an earlier enumeration win a tie. Ordering has
        // to be total: two runs on one host that pick different devices is a
        // performance mystery nobody can reproduce.
        .max_by_key(|c| (c.class.rank(), usize::MAX - c.index))
        .copied()
        .ok_or_else(|| NoDevice {
            enumerated: candidates.len(),
            floors: candidates
                .iter()
                .filter_map(|c| c.verdict.as_ref().err().copied())
                .collect(),
        })
}

/// Why the instance could not be created.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenFailure {
    /// The Vulkan loader is absent or refused to initialize.
    NoLoader,
    /// The loader supports an API below this rail's baseline, so no device it
    /// enumerates could clear the floor.
    LoaderBelowBaseline {
        reported: ApiVersion,
    },
    CreateInstance {
        result: vk::Result,
    },
    EnumerateDevices {
        result: vk::Result,
    },
    NoDevice(NoDevice),
}

impl OpenFailure {
    #[must_use]
    pub const fn slug(&self) -> &'static str {
        match self {
            Self::NoLoader => "vk_host_no_loader",
            Self::LoaderBelowBaseline { .. } => "vk_host_loader_below_baseline",
            Self::CreateInstance { .. } => "vk_host_create_instance",
            Self::EnumerateDevices { .. } => "vk_host_enumerate_devices",
            Self::NoDevice(_) => "vk_host_no_usable_device",
        }
    }
}

impl std::fmt::Display for OpenFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoLoader => f.write_str(self.slug()),
            Self::LoaderBelowBaseline { reported } => {
                write!(f, "{} reported={reported}", self.slug())
            }
            Self::CreateInstance { result } | Self::EnumerateDevices { result } => {
                write!(f, "{} result={result:?}", self.slug())
            }
            Self::NoDevice(inner) => write!(f, "{inner}"),
        }
    }
}

/// The immutable host: instance, selected physical device, and its catalog.
///
/// Not `Clone`. One per process, and its `Drop` destroys the instance, so a
/// second copy would be a second destruction of one handle.
pub struct VulkanHost {
    // Held because the instance's function pointers are loaded through it and
    // unloading the library while they are live is a use-after-free. Never
    // read.
    _entry: ash::Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    class: DeviceClass,
    census: Census,
}

impl VulkanHost {
    /// Create the instance, enumerate, judge, and bind one device.
    ///
    /// # Errors
    ///
    /// [`OpenFailure`] for a missing loader, a refused instance, or a host
    /// where nothing clears the floor.
    pub fn open(application: &str) -> Result<Self, OpenFailure> {
        // SAFETY: `Entry::load` dlopens the Vulkan loader. The library stays
        // loaded for as long as the returned `Entry` lives, and `Self` holds it
        // beside every function table taken from it.
        let entry = unsafe { ash::Entry::load() }.map_err(|_| OpenFailure::NoLoader)?;

        // SAFETY: the entry's loader is live for the whole of this function.
        let loader_version = match unsafe { entry.try_enumerate_instance_version() } {
            // A 1.0 loader reports `None`, and 1.2 is the baseline.
            Ok(None) => {
                return Err(OpenFailure::LoaderBelowBaseline {
                    reported: ApiVersion { major: 1, minor: 0 },
                })
            }
            Ok(Some(packed)) => ApiVersion::decode(packed),
            Err(result) => return Err(OpenFailure::CreateInstance { result }),
        };
        if !loader_version.at_least(1, 2) {
            return Err(OpenFailure::LoaderBelowBaseline {
                reported: loader_version,
            });
        }

        let name = CString::new(application).unwrap_or_else(|_| c"reims-vgpu".to_owned());
        // Ask for the baseline and no more. A higher `apiVersion` here does not
        // enable anything — every capability above 1.2 is gated on the census —
        // and asking for one the loader lacks is an instance-creation failure
        // for no gain.
        let app = vk::ApplicationInfo::default()
            .application_name(&name)
            .api_version(vk::API_VERSION_1_2);
        let create = vk::InstanceCreateInfo::default().application_info(&app);

        // SAFETY: `create` and everything it points at outlive the call.
        let instance = unsafe { entry.create_instance(&create, None) }
            .map_err(|result| OpenFailure::CreateInstance { result })?;

        match Self::choose(&instance) {
            Ok((physical, class, census)) => Ok(Self {
                _entry: entry,
                instance,
                physical,
                class,
                census,
            }),
            Err(failure) => {
                // Nothing took ownership of the instance, and `Drop` runs only
                // on the success path above — so this is the one place that
                // would otherwise leak it.
                //
                // SAFETY: no device was created from this instance and no child
                // object exists, which is exactly the state a failed selection
                // leaves it in.
                unsafe { instance.destroy_instance(None) };
                Err(failure)
            }
        }
    }

    /// Enumerate, judge, and select. Creates nothing, so a failure here leaves
    /// the instance the caller owns as its only cleanup.
    fn choose(
        instance: &ash::Instance,
    ) -> Result<(vk::PhysicalDevice, DeviceClass, Census), OpenFailure> {
        // SAFETY: the instance outlives this call.
        let devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|result| OpenFailure::EnumerateDevices { result })?;

        let judged: Vec<(vk::PhysicalDevice, Candidate)> = devices
            .iter()
            .copied()
            .enumerate()
            .map(|(index, physical)| {
                // SAFETY: `physical` came from this instance's enumeration.
                let (class, verdict) = unsafe { judge(instance, physical) };
                (
                    physical,
                    Candidate {
                        index,
                        class,
                        verdict,
                    },
                )
            })
            .collect();

        let candidates: Vec<Candidate> = judged.iter().map(|(_, c)| *c).collect();
        let chosen = select(&candidates).map_err(OpenFailure::NoDevice)?;
        let (physical, _) = judged[chosen.index];
        let census = chosen.verdict.expect("a selected candidate was admitted");
        Ok((physical, chosen.class, census))
    }

    #[must_use]
    pub const fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    #[must_use]
    pub const fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical
    }

    /// The catalog every capability decision in this crate reads from.
    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    /// One line recording the bound device's capability census, as the
    /// architecture asks for with every run.
    #[must_use]
    pub fn report_line(&self) -> String {
        format!("{} class={}", self.census.report_line(), self.class.slug())
    }
}

impl Drop for VulkanHost {
    fn drop(&mut self) {
        // SAFETY: this type owns the instance, is not `Clone`, and creates no
        // device or other child object from it — so nothing derived from the
        // instance can outlive this call.
        unsafe { self.instance.destroy_instance(None) };
    }
}

impl std::fmt::Debug for VulkanHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanHost")
            .field("class", &self.class)
            .field("census", &self.census)
            .finish_non_exhaustive()
    }
}

/// Read one physical device and hand it to the census.
///
/// # Safety
///
/// `physical` must have come from `instance`'s enumeration.
unsafe fn judge(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> (DeviceClass, Result<Census, Floor>) {
    let properties = unsafe { instance.get_physical_device_properties(physical) };
    let class = DeviceClass::of(properties.device_type);

    let extension_properties =
        unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
    let names: Vec<String> = extension_properties
        .iter()
        .filter_map(|e| {
            // SAFETY: the driver fills `extension_name` with a NUL-terminated
            // name; the array is the storage it was written into.
            unsafe { CStr::from_ptr(e.extension_name.as_ptr()) }
                .to_str()
                .ok()
                .map(str::to_owned)
        })
        .collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let has = |name: &str| names.contains(&name);

    // The feature chain, built from what was enumerated and never from what
    // was hoped for. See the module doc.
    let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
    let mut descriptor_buffer = vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default();
    let mut synchronization2 = vk::PhysicalDeviceSynchronization2Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut vulkan12);
    if has(extension::MESH_SHADER) {
        features = features.push_next(&mut mesh);
    }
    if has(extension::DESCRIPTOR_BUFFER) {
        features = features.push_next(&mut descriptor_buffer);
    }
    if has(extension::SYNCHRONIZATION_2) {
        features = features.push_next(&mut synchronization2);
    }
    unsafe { instance.get_physical_device_features2(physical, &mut features) };

    let mut push = vk::PhysicalDevicePushDescriptorPropertiesKHR::default();
    if has(extension::PUSH_DESCRIPTOR) {
        let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut push);
        unsafe { instance.get_physical_device_properties2(physical, &mut properties2) };
    }

    let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical) };

    let verdict = Census::take(Reported {
        api_version: properties.api_version,
        extensions: &names,
        timeline_semaphore: vulkan12.timeline_semaphore == vk::TRUE,
        synchronization2: synchronization2.synchronization2 == vk::TRUE,
        mesh_shader: mesh.mesh_shader == vk::TRUE,
        descriptor_buffer: descriptor_buffer.descriptor_buffer == vk::TRUE,
        max_push_descriptors: push.max_push_descriptors,
        memory: &memory,
        queue_families: &queue_families,
    });
    (class, verdict)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::census::DescriptorBufferProbe;

    fn admitted(index: usize, class: DeviceClass) -> Candidate {
        let memory = crate::memory::fixtures::nvidia_discrete();
        let families = [vk::QueueFamilyProperties {
            queue_flags: vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            queue_count: 1,
            timestamp_valid_bits: 64,
            min_image_transfer_granularity: vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
        }];
        let verdict = Census::take(Reported {
            api_version: vk::make_api_version(0, 1, 2, 0),
            extensions: &[extension::SWAPCHAIN],
            timeline_semaphore: true,
            synchronization2: false,
            mesh_shader: false,
            descriptor_buffer: false,
            max_push_descriptors: 0,
            memory: &memory,
            queue_families: &families,
        });
        assert!(verdict.is_ok());
        Candidate {
            index,
            class,
            verdict,
        }
    }

    fn refused(index: usize, class: DeviceClass, floor: Floor) -> Candidate {
        Candidate {
            index,
            class,
            verdict: Err(floor),
        }
    }

    #[test]
    fn a_software_rasterizer_that_enumerated_first_does_not_win() {
        let chosen = select(&[
            admitted(0, DeviceClass::Cpu),
            admitted(1, DeviceClass::Discrete),
        ])
        .expect("both are usable");
        assert_eq!(chosen.index, 1);
        assert_eq!(chosen.class, DeviceClass::Discrete);
    }

    #[test]
    fn a_hybrid_host_prefers_the_discrete_part_wherever_it_enumerated() {
        for (discrete, integrated) in [(0usize, 1usize), (1, 0)] {
            let mut candidates = vec![
                admitted(discrete, DeviceClass::Discrete),
                admitted(integrated, DeviceClass::Integrated),
            ];
            candidates.sort_by_key(|c| c.index);
            let chosen = select(&candidates).expect("both are usable");
            assert_eq!(chosen.index, discrete);
        }
    }

    #[test]
    fn a_rank_tie_keeps_the_first_enumerated_device() {
        let chosen = select(&[
            admitted(0, DeviceClass::Discrete),
            admitted(1, DeviceClass::Discrete),
            admitted(2, DeviceClass::Discrete),
        ])
        .expect("all usable");
        assert_eq!(chosen.index, 0, "selection has to be reproducible");
    }

    /// The claim that keeps preference from becoming admission: a better class
    /// that failed the floor loses to a worse class that cleared it.
    #[test]
    fn a_class_can_never_admit_a_device_the_floor_refused() {
        let chosen = select(&[
            refused(
                0,
                DeviceClass::Discrete,
                Floor::NoTimelineSemaphores {
                    reported: ApiVersion { major: 1, minor: 3 },
                },
            ),
            admitted(1, DeviceClass::Cpu),
        ])
        .expect("the CPU device cleared the floor");
        assert_eq!(chosen.class, DeviceClass::Cpu);
    }

    #[test]
    fn a_host_where_nothing_clears_the_floor_names_every_reason() {
        let failure = select(&[
            refused(
                0,
                DeviceClass::Discrete,
                Floor::ApiTooOld {
                    reported: ApiVersion { major: 1, minor: 1 },
                },
            ),
            refused(1, DeviceClass::Integrated, Floor::NoSwapchain),
        ])
        .expect_err("nothing usable");
        assert_eq!(failure.enumerated, 2);
        assert_eq!(failure.floors.len(), 2);
        let line = failure.to_string();
        assert!(line.contains("vk_census_api_too_old"), "{line}");
        assert!(line.contains("vk_census_no_swapchain"), "{line}");
    }

    #[test]
    fn a_host_with_no_devices_at_all_is_its_own_reading() {
        let failure = select(&[]).expect_err("nothing enumerated");
        assert_eq!(failure.enumerated, 0);
        assert!(failure.floors.is_empty());
    }

    #[test]
    fn the_class_order_is_the_one_documented() {
        let mut classes = [
            DeviceClass::Cpu,
            DeviceClass::Discrete,
            DeviceClass::Other,
            DeviceClass::Virtual,
            DeviceClass::Integrated,
        ];
        classes.sort_by_key(|c| std::cmp::Reverse(c.rank()));
        assert_eq!(
            classes,
            [
                DeviceClass::Discrete,
                DeviceClass::Integrated,
                DeviceClass::Virtual,
                DeviceClass::Other,
                DeviceClass::Cpu,
            ]
        );
        assert_eq!(
            DeviceClass::of(vk::PhysicalDeviceType::DISCRETE_GPU),
            DeviceClass::Discrete
        );
        // Anything the enum grows later reads as `Other` rather than as a
        // panic or as the first arm.
        assert_eq!(
            DeviceClass::of(vk::PhysicalDeviceType::from_raw(999)),
            DeviceClass::Other
        );
    }

    /// Bring up the real loader on whatever device this machine has.
    ///
    /// Every other test here is pure, so this one is about the parts that
    /// cannot be: that the feature chain is accepted by a real driver, that the
    /// census reads consistently off it, and that the instance is created and
    /// destroyed without leaking. A host with no Vulkan says so and asserts
    /// nothing further — but it says which of the two it did, so a silently
    /// skipped run is distinguishable from a passing one in the output.
    #[test]
    fn a_real_host_is_judged_and_its_census_agrees_with_itself() {
        match VulkanHost::open("reims-vgpu-vulkan test") {
            Ok(host) => {
                let line = host.report_line();
                println!("real device: {line}");
                assert!(!line.contains("name="), "the census names no device");

                let census = host.census();
                assert!(census.api().at_least(1, 2), "admitted below the baseline");
                assert!(census.can_present());

                // A census is one snapshot: asking twice cannot disagree.
                assert_eq!(census, host.census());
                assert_eq!(census.report_line(), host.census().report_line());

                // The projections are of the same snapshot.
                assert_eq!(census.host_cell().topology, census.memory().topology);
                assert!(census.queues().universal().graphics());

                // An unmeasured device never reaches the top descriptor rung,
                // whatever it reports.
                assert!(!census.descriptors().descriptor_buffer_qualified);
                let probed = census.with_descriptor_buffer_probe(DescriptorBufferProbe::Passed);
                assert_eq!(
                    probed.descriptors().descriptor_buffer_qualified,
                    census.descriptors().descriptor_buffer,
                    "a probe qualifies exactly what the device reported"
                );
            }
            Err(failure) => {
                println!("no real device: {failure}");
                // Not an assertion about this machine — a host with no loader
                // is a legal machine to run the pure tests on. What is asserted
                // is that the refusal is typed and names itself.
                assert!(!failure.slug().is_empty());
            }
        }
    }
}
