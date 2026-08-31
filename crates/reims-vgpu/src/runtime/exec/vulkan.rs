//! Work this rail does before a submission may be consumed.
//!
//! Cold AIR translation is immutable CPU work with no protocol ownership, and
//! this rail wants it done before the packet is taken: a stream whose render or
//! compute pipelines are still translating is deferred rather than executed, so
//! a replay cannot duplicate clears, fences, dispatches or guest writeback.
//!
//! A rail that translates nothing has nothing to preflight, which is why the
//! whole of this is reached through
//! [`crate::backend::Backend::preflight_translations`] and `runtime::exec`
//! never names it.

use super::*;

pub(crate) fn preflight_render_translations<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    stream: &[u8],
) -> bool {
    use crate::runtime::drain::{note_preflight_part, note_preflight_pipe, PreflightPart};
    let refs_started = std::time::Instant::now();
    let pipelines = render_pipeline_refs(stream);
    note_preflight_part(
        PreflightPart::Refs,
        refs_started.elapsed().as_nanos() as u64,
    );
    let mut pending = false;
    for pipeline_ref in pipelines {
        note_preflight_pipe();
        // The draw path's own memo already knows whether these two shaders are
        // translated, and answers for ~0.6 us against the 4.3 us of guest
        // resolves below. `translations_ready` states why that is not a weaker
        // answer — chiefly that the translate cache never evicts, so a shader
        // this memo saw translated is still translated.
        if crate::runtime::pipeline_resolve::translations_ready(state, host, task_id, pipeline_ref)
        {
            continue;
        }
        let air_started = std::time::Instant::now();
        // The MTLB containers, not owned copies of the AIR inside them: the two
        // `ensure_cached_async` calls below borrow, digest and drop, so copying
        // first would allocate twice per pipeline ref for bytes nothing keeps.
        let pair = draw::vulkan::load_render_mtlb_pair(state, host, task_id, pipeline_ref);
        note_preflight_part(PreflightPart::Air, air_started.elapsed().as_nanos() as u64);
        let Ok((v_mtlb, f_mtlb)) = pair else {
            // Normal execution emits the precise pipeline/MTLB failure. A
            // missing plan input is deterministic, not asynchronous work.
            continue;
        };
        // A container whose AIR will not extract is the same "deterministic
        // missing plan input" as one that would not load: normal execution
        // reports it precisely, and there is no asynchronous work to await.
        let (Ok(v_air), Ok(f_air)) = (
            crate::runtime::mtlb::extract_air(&v_mtlb),
            crate::runtime::mtlb::extract_air(&f_mtlb),
        ) else {
            continue;
        };
        let cache_started = std::time::Instant::now();
        if !crate::runtime::m2v_cache::ensure_cached_async(
            v_air,
            metal2vulkan::passes::Stage::Vertex,
            pipeline_ref,
        ) {
            pending = true;
        }
        if !crate::runtime::m2v_cache::ensure_cached_async(
            f_air,
            metal2vulkan::passes::Stage::Fragment,
            pipeline_ref,
        ) {
            pending = true;
        }
        note_preflight_part(
            PreflightPart::Cache,
            cache_started.elapsed().as_nanos() as u64,
        );
    }
    pending
}

pub(crate) fn render_pipeline_refs(stream: &[u8]) -> Vec<u32> {
    // Deliberately silent on a framing refusal: this is a speculative pre-scan of
    // the very stream `walk_stream` is about to frame and report on. Logging here
    // would double every `stream_frame_fail` line for no added information.
    let Ok(segs) = stream::iter_segments(stream) else {
        return Vec::new();
    };
    let mut pipelines = Vec::new();
    for seg in segs {
        if seg.type_ != SEGMENT_TYPE_RENDER {
            continue;
        }
        let mut cursor = 0usize;
        let mut next = decode_first_record(stream, &seg, &mut cursor);
        while let Ok(rec) = next {
            let start = rec.bytes_offset as usize;
            let end = start.saturating_add(rec.length as usize);
            if let Some(bytes) = stream.get(start..end) {
                if let Ok(cmd) = render::decode(bytes) {
                    if cmd.kind == RenderKind::SetPipeline
                        && cmd.pipeline_ref != 0
                        && !pipelines.contains(&cmd.pipeline_ref)
                    {
                        pipelines.push(cmd.pipeline_ref);
                    }
                }
            }
            next = decode_next_record(stream, &seg, &mut cursor);
        }
    }

    pipelines
}

pub(crate) fn preflight_compute_translations<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    stream: &[u8],
) -> bool {
    use crate::runtime::drain::{note_preflight_part, note_preflight_pipe, PreflightPart};
    let refs_started = std::time::Instant::now();
    let inputs = compute_translation_inputs(stream);
    note_preflight_part(
        PreflightPart::Refs,
        refs_started.elapsed().as_nanos() as u64,
    );
    let mut pending = false;
    for (pipeline_ref, local_size) in inputs {
        note_preflight_pipe();
        let air_started = std::time::Instant::now();
        let loaded = compute_exec::load_compute_pipeline(state, host, task_id, pipeline_ref)
            .and_then(|pipeline| {
                crate::runtime::mtlb::load_mtlb(
                    state,
                    host,
                    task_id,
                    pipeline.kernel_func_ref,
                    crate::runtime::mtlb::AirLoadRail::Compute,
                )
            });
        note_preflight_part(PreflightPart::Air, air_started.elapsed().as_nanos() as u64);
        let Some(mtlb) = loaded else {
            continue;
        };
        let Ok(air) = crate::runtime::mtlb::extract_air(&mtlb) else {
            continue;
        };
        let cache_started = std::time::Instant::now();
        let cached =
            crate::runtime::m2v_cache::ensure_cached_kernel_async(air, local_size, pipeline_ref);
        note_preflight_part(
            PreflightPart::Cache,
            cache_started.elapsed().as_nanos() as u64,
        );
        if !cached {
            pending = true;
        }
    }
    pending
}

/// Structurally collect compute pipeline + LocalSize pairs in command order.
/// Threads-indirect carries LocalSize in guest argument memory rather than the
/// stream record, so it deliberately remains on the synchronous fallback.
pub(crate) fn compute_translation_inputs(stream: &[u8]) -> Vec<(u32, [u32; 3])> {
    // Silent for the same reason as `render_pipeline_refs`: a pre-scan whose
    // framing refusal `walk_stream` will report once, with the task attached.
    let Ok(segs) = stream::iter_segments(stream) else {
        return Vec::new();
    };
    let mut inputs = Vec::new();
    for seg in segs {
        if seg.type_ != SEGMENT_TYPE_COMPUTE {
            continue;
        }
        let mut pipeline_ref = 0u32;
        let mut cursor = 0usize;
        let mut next = decode_first_record(stream, &seg, &mut cursor);
        while let Ok(rec) = next {
            let start = rec.bytes_offset as usize;
            let end = start.saturating_add(rec.length as usize);
            if let Some(bytes) = stream.get(start..end) {
                if let Ok(cmd) = compute::decode(bytes) {
                    match cmd.kind {
                        ComputeKind::Pipeline => pipeline_ref = cmd.pipeline_ref,
                        ComputeKind::DispatchThreadgroups
                        | ComputeKind::DispatchThreadgroupsIndirect
                        | ComputeKind::DispatchThreads => {
                            let dims = cmd.threads_per_threadgroup;
                            let local_size = [
                                u32::try_from(dims.x).ok(),
                                u32::try_from(dims.y).ok(),
                                u32::try_from(dims.z).ok(),
                            ];
                            if pipeline_ref != 0 {
                                if let [Some(x), Some(y), Some(z)] = local_size {
                                    let item = (pipeline_ref, [x, y, z]);
                                    if x != 0 && y != 0 && z != 0 && !inputs.contains(&item) {
                                        inputs.push(item);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            next = decode_next_record(stream, &seg, &mut cursor);
        }
    }
    inputs
}
/// [`crate::backend::Backend::preflight_translations`] for this rail.
///
/// Every stream is scanned, not just up to the first pending one: the point is
/// to *start* every cold translation this packet needs, so they proceed in
/// parallel and the packet is retried once rather than once per stream.
pub fn preflight_translations<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    streams: &[Vec<u8>],
) -> bool {
    streams.iter().fold(false, |pending, stream| {
        let render_pending = preflight_render_translations(state, host, task_id, stream);
        let compute_pending = preflight_compute_translations(state, host, task_id, stream);
        render_pending || compute_pending || pending
    })
}
