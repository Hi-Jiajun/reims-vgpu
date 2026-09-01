//! The `VkDevice` and everything that dies with it.
//!
//! # What an epoch is, and why it is a boundary rather than a field
//!
//! A `VkDevice` can be lost, and when it is, every handle made from it becomes
//! unusable at the same instant: queues, timeline semaphores, command pools,
//! descriptor pools, allocations, pipelines. The architecture makes that set an
//! object — the device epoch — precisely so no code has to remember which
//! handles those were. There is one active epoch per live session, and an old
//! one is retained only until the host work it already issued is retired or
//! abandoned.
//!
//! Guest semantic validity and Vulkan handle validity are different lifetimes,
//! and a native lease carries both: a session generation says whether the guest
//! still means the object, and [`EpochId`] says whether its handles may still
//! be touched. Collapsing them makes a guest reset destroy live GPU work, or a
//! device loss leave semantic state claiming to own dead handles.
//!
//! # The enabled set is derived from the census and from nothing else
//!
//! [`Enabled::for_census`] is the one place a feature or extension is asked
//! for, and it can only ask for what [`Census::take`] already admitted. That is
//! the difference between a rail that uses a capability and one that hopes for
//! it: enabling a feature the device did not report is undefined behaviour that
//! usually works, right up to the driver where it does not.
//!
//! The floor is enabled unconditionally because a device that could not supply
//! it never became a census in the first place.
//!
//! # This module creates the device and stops
//!
//! It owns no command pool, no descriptor pool, no allocation and no pipeline.
//! Those are per-worker or per-owner and are handed the device rather than
//! living in it — see [`crate::pools`] and [`crate::descriptor`]. What is here
//! is what device creation itself decides: which features are on, which
//! extensions are requested, and which queues exist.

use crate::census::{Census, DeviceExtensions};
use crate::queues::QueuePlan;
use ash::vk;
use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};

/// Which incarnation of the device a handle belongs to.
///
/// Monotone and never reused within a process, so a stale lease naming an
/// earlier epoch is recognisably stale rather than accidentally valid against a
/// recreated device that happens to sit at the same address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EpochId(u64);

impl EpochId {
    fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        // Relaxed: the only requirement is uniqueness, and every ordering
        // relationship an epoch takes part in is carried by the owner that
        // hands it out rather than by this counter.
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for EpochId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "epoch{}", self.0)
    }
}

/// What device creation turns on.
///
/// Derived from a [`Census`] and never assembled by hand, so nothing can be
/// enabled that was not admitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Enabled {
    pub extensions: DeviceExtensions,
    /// Always true. It is the support floor, and a device that lacked it never
    /// became a census.
    pub timeline_semaphore: bool,
    pub synchronization2: bool,
    pub mesh_shader: bool,
    pub descriptor_buffer: bool,
}

impl Enabled {
    /// Everything the census admitted, and nothing else.
    #[must_use]
    pub fn for_census(census: &Census) -> Self {
        Self {
            extensions: census.extensions(),
            timeline_semaphore: true,
            // `synchronization2` as a *feature* is only chained below 1.3; at
            // 1.3 and above the capability is core and there is no feature
            // struct to set. The census already collapsed those two routes into
            // one answer, and the extension list says which route this is.
            synchronization2: census.synchronization2() && census.extensions().synchronization2,
            mesh_shader: census.stages().mesh_shader,
            descriptor_buffer: census.descriptors().descriptor_buffer,
        }
    }
}

/// Why the device could not be created.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceFailure {
    CreateDevice {
        result: vk::Result,
    },
    /// The chosen family reported queues and then handed none out. A driver
    /// contradiction rather than a capability answer, so it is its own reason.
    NoQueue {
        family: u32,
    },
}

impl DeviceFailure {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::CreateDevice { .. } => "vk_device_create",
            Self::NoQueue { .. } => "vk_device_no_queue",
        }
    }
}

impl std::fmt::Display for DeviceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreateDevice { result } => write!(f, "{} result={result:?}", self.slug()),
            Self::NoQueue { family } => write!(f, "{} family={family}", self.slug()),
        }
    }
}

/// One `VkDevice`, its identity, and the queue ledger for it.
///
/// Not `Clone`: its `Drop` destroys the device, and a second copy would be a
/// second destruction. Everything made from the device has to be retired before
/// it is dropped, which is what [`Self::wait_idle`] is for.
pub struct DeviceEpoch {
    id: EpochId,
    device: ash::Device,
    census: Census,
    enabled: Enabled,
    queues: QueuePlan,
}

impl DeviceEpoch {
    /// Create the device from a host's catalog.
    ///
    /// One queue is requested from the chosen family. More would be a
    /// concurrency decision, and the architecture puts that behind measurement
    /// rather than behind "the family reported a count".
    ///
    /// # Errors
    ///
    /// [`DeviceFailure`] when the driver refuses creation or hands back no
    /// queue.
    pub fn create(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        census: Census,
    ) -> Result<Self, DeviceFailure> {
        let enabled = Enabled::for_census(&census);
        let family = census.queues().universal().index;

        let names: Vec<CString> = enabled
            .extensions
            .names()
            .into_iter()
            .map(|n| CString::new(n).expect("an extension name has no interior NUL"))
            .collect();
        let pointers: Vec<*const i8> = names.iter().map(|n| n.as_ptr()).collect();

        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];

        // Chained only where the capability was admitted, for the reason the
        // census module gives: a feature struct for something the driver never
        // reported is a question it has no answer to.
        let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
        let mut synchronization2 =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let mut mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default().mesh_shader(true);
        let mut descriptor_buffer =
            vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default().descriptor_buffer(true);

        let mut create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&pointers)
            .push_next(&mut vulkan12);
        if enabled.synchronization2 {
            create = create.push_next(&mut synchronization2);
        }
        if enabled.mesh_shader {
            create = create.push_next(&mut mesh);
        }
        if enabled.descriptor_buffer {
            create = create.push_next(&mut descriptor_buffer);
        }

        // SAFETY: `physical` belongs to `instance`, and every structure and
        // array `create` points at outlives the call.
        let device = unsafe { instance.create_device(physical, &create, None) }
            .map_err(|result| DeviceFailure::CreateDevice { result })?;

        // SAFETY: the family and index were requested above, so the queue
        // exists. A driver that returns a null handle here is the
        // contradiction `NoQueue` names.
        let queue = unsafe { device.get_device_queue(family, 0) };
        if queue == vk::Queue::null() {
            // SAFETY: nothing was created from the device.
            unsafe { device.destroy_device(None) };
            return Err(DeviceFailure::NoQueue { family });
        }

        Ok(Self {
            id: EpochId::next(),
            device,
            census,
            enabled,
            queues: QueuePlan::adopt(census.queues()),
        })
    }

    #[must_use]
    pub const fn id(&self) -> EpochId {
        self.id
    }

    #[must_use]
    pub const fn device(&self) -> &ash::Device {
        &self.device
    }

    /// The catalog this epoch was created against.
    ///
    /// A copy of the host's, not a second reading: a decision made here and a
    /// decision made at selection see one snapshot.
    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    #[must_use]
    pub const fn enabled(&self) -> &Enabled {
        &self.enabled
    }

    /// The ledger of which of this device's queues have an owner.
    ///
    /// `&mut` because claiming is the only thing anyone does with it, and two
    /// claimants would be two owners of one externally-synchronized `VkQueue`.
    pub fn queues(&mut self) -> &mut QueuePlan {
        &mut self.queues
    }

    /// Wait for every submission on this device to finish.
    ///
    /// Only correct at epoch shutdown, after ingress and submissions have
    /// stopped. Routine destruction never comes here: it goes through timeline
    /// retirement, because a device-wide idle in a resource path serialises
    /// every worker behind the slowest one.
    ///
    /// # Errors
    ///
    /// The driver's result, including `VK_ERROR_DEVICE_LOST` — which is not a
    /// reason to skip destruction, since the handles still have to be given
    /// back.
    pub fn wait_idle(&self) -> Result<(), vk::Result> {
        // SAFETY: the device is live and this owner is the only one that
        // submits, so no queue is being used concurrently.
        unsafe { self.device.device_wait_idle() }
    }
}

impl Drop for DeviceEpoch {
    fn drop(&mut self) {
        // The architecture's rule: a final clean epoch shutdown may idle the
        // device, and only after ingress and submissions have stopped. A loss
        // makes the wait fail and destruction still has to happen — Vulkan's
        // finite-return rules mean the handles are ours to give back either
        // way.
        let _ = self.wait_idle();
        // SAFETY: this type owns the device, is not `Clone`, and the wait above
        // has completed every submission that could still name a child object.
        unsafe { self.device.destroy_device(None) };
    }
}

impl std::fmt::Debug for DeviceEpoch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceEpoch")
            .field("id", &self.id)
            .field("enabled", &self.enabled)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::census::{extension, ApiVersion, Reported};
    use crate::host::VulkanHost;
    use crate::memory::fixtures as mem;

    fn families() -> [vk::QueueFamilyProperties; 1] {
        [vk::QueueFamilyProperties {
            queue_flags: vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            queue_count: 4,
            timestamp_valid_bits: 64,
            min_image_transfer_granularity: vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
        }]
    }

    fn census_of(api: (u32, u32), extensions: &[&str], features: (bool, bool, bool)) -> Census {
        let memory = mem::apple_m3_max();
        let families = families();
        Census::take(Reported {
            api_version: vk::make_api_version(0, api.0, api.1, 0),
            extensions,
            timeline_semaphore: true,
            synchronization2: features.0,
            mesh_shader: features.1,
            descriptor_buffer: features.2,
            max_push_descriptors: 32,
            memory: &memory,
            queue_families: &families,
        })
        .expect("admitted")
    }

    #[test]
    fn the_floor_is_always_enabled() {
        let bare = census_of((1, 2), &[extension::SWAPCHAIN], (false, false, false));
        let enabled = Enabled::for_census(&bare);
        assert!(enabled.timeline_semaphore);
        assert!(!enabled.synchronization2);
        assert!(!enabled.mesh_shader);
        assert!(!enabled.descriptor_buffer);
        assert_eq!(enabled.extensions.names(), vec![extension::SWAPCHAIN]);
    }

    /// The claim that keeps this from being a wish list.
    #[test]
    fn nothing_is_enabled_that_the_census_did_not_admit() {
        // Every combination of enumerated-extension and reported-feature, at
        // three API versions.
        let mut checked = 0;
        for (major, minor) in [(1u32, 2u32), (1, 3), (1, 4)] {
            for mesh_ext in [false, true] {
                for mesh_feature in [false, true] {
                    for buffer_ext in [false, true] {
                        for buffer_feature in [false, true] {
                            let mut names = vec![extension::SWAPCHAIN];
                            if mesh_ext {
                                names.push(extension::MESH_SHADER);
                            }
                            if buffer_ext {
                                names.push(extension::DESCRIPTOR_BUFFER);
                            }
                            let census = census_of(
                                (major, minor),
                                &names,
                                (false, mesh_feature, buffer_feature),
                            );
                            let enabled = Enabled::for_census(&census);

                            assert_eq!(enabled.mesh_shader, mesh_ext && mesh_feature);
                            assert_eq!(enabled.descriptor_buffer, buffer_ext && buffer_feature);
                            // And the extension list agrees with the features.
                            assert_eq!(enabled.extensions.mesh_shader, enabled.mesh_shader);
                            assert_eq!(
                                enabled.extensions.descriptor_buffer,
                                enabled.descriptor_buffer
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(checked, 48);
        // Non-vacuity: the sweep contained a case where each was on.
        let both = census_of(
            (1, 3),
            &[
                extension::SWAPCHAIN,
                extension::MESH_SHADER,
                extension::DESCRIPTOR_BUFFER,
            ],
            (false, true, true),
        );
        let enabled = Enabled::for_census(&both);
        assert!(enabled.mesh_shader && enabled.descriptor_buffer);
    }

    /// A capability that arrived through core has no feature struct to set and
    /// no name to request, and the enabled set says so.
    #[test]
    fn a_core_capability_is_enabled_through_neither_a_name_nor_a_feature() {
        let promoted = census_of((1, 3), &[extension::SWAPCHAIN], (false, false, false));
        assert!(promoted.synchronization2(), "core at 1.3");
        let enabled = Enabled::for_census(&promoted);
        assert!(
            !enabled.synchronization2,
            "there is no feature struct to chain for a core capability"
        );
        assert!(!enabled.extensions.synchronization2);

        // Through the extension below its promotion, both are set.
        let extended = census_of(
            (1, 2),
            &[extension::SWAPCHAIN, extension::SYNCHRONIZATION_2],
            (true, false, false),
        );
        let enabled = Enabled::for_census(&extended);
        assert!(enabled.synchronization2);
        assert!(enabled.extensions.synchronization2);
    }

    #[test]
    fn an_epoch_id_is_unique_and_names_itself() {
        let a = EpochId::next();
        let b = EpochId::next();
        assert_ne!(a, b);
        assert!(b > a, "an id has to say which incarnation is later");
        assert!(a.to_string().starts_with("epoch"));
    }

    #[test]
    fn a_failure_names_itself() {
        let refused = DeviceFailure::CreateDevice {
            result: vk::Result::ERROR_INITIALIZATION_FAILED,
        };
        assert_eq!(refused.slug(), "vk_device_create");
        assert!(refused.to_string().contains("INITIALIZATION"));
        assert!(DeviceFailure::NoQueue { family: 3 }
            .to_string()
            .contains("family=3"));
    }

    /// Bring up a real device: the part purity cannot reach is whether a driver
    /// accepts the exact feature chain and extension list this census produced.
    /// Every combination that reaches `create` is one the census admitted, so a
    /// refusal here would mean the admission rule is wrong.
    #[test]
    fn a_real_device_accepts_the_set_its_census_admitted() {
        let Ok(host) = VulkanHost::open("reims-vgpu-vulkan device test") else {
            println!("no real device: nothing to create");
            return;
        };
        let census = host.census();
        let epoch = DeviceEpoch::create(host.instance(), host.physical_device(), census)
            .expect("the driver refused a set its own census admitted");

        println!("real device: {} {:?}", epoch.id(), epoch.enabled());
        assert_eq!(epoch.census(), census, "one snapshot, not a second reading");
        assert!(epoch.enabled().timeline_semaphore);

        // A second epoch on the same device is a distinct incarnation.
        let second = DeviceEpoch::create(host.instance(), host.physical_device(), census)
            .expect("a second device");
        assert_ne!(epoch.id(), second.id());

        // The queue ledger is this epoch's, and hands each queue out once.
        let mut epoch = epoch;
        let family = census.queues().universal().index;
        let owned = epoch.queues().claim_in(family, 0).expect("queue zero");
        assert!(
            epoch.queues().claim_in(family, 0).is_none(),
            "a VkQueue has exactly one owner"
        );
        epoch.queues().release(owned);

        assert_eq!(epoch.wait_idle(), Ok(()));
        drop(second);
        drop(epoch);
    }

    #[test]
    fn a_below_floor_version_never_reaches_device_creation() {
        // Not a device test: it asserts that the type this module takes can
        // only have come from an admitted census, so `create` has no
        // below-floor case to handle.
        let memory = mem::apple_m3_max();
        let families = families();
        let refused = Census::take(Reported {
            api_version: vk::make_api_version(0, 1, 1, 0),
            extensions: &[extension::SWAPCHAIN],
            timeline_semaphore: true,
            synchronization2: false,
            mesh_shader: false,
            descriptor_buffer: false,
            max_push_descriptors: 0,
            memory: &memory,
            queue_families: &families,
        });
        assert_eq!(
            refused,
            Err(crate::census::Floor::ApiTooOld {
                reported: ApiVersion { major: 1, minor: 1 }
            })
        );
    }
}
