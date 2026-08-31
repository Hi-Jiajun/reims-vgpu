//! The census lines only the Metal rail can answer.
//!
//! One line: this rail's compiled-object cache levels. Its Vulkan counterpart
//! asks the same question of different tables, which is why both are reached
//! through [`crate::backend::Backend::emit_census`] rather than through one
//! `cfg`-selected function — a build carrying both rails has to print the one
//! belonging to the device it is actually running.
//!
//! It lives here and not beside the drain's neutral census for the same reason:
//! the only thing it reads is this rail's own cache, and a reporting module in
//! the neutral runtime forced that cache to be re-exported out of the rail so a
//! stranger could count it.

/// This rail's compiled-object cache levels, as **levels** rather than
/// per-window deltas — the same question and cadence as its Vulkan
/// counterpart, over different tables. This arm builds `MTLFunction` / `MTLRenderPipelineState` /
/// `MTLComputePipelineState` / `MTLSamplerState` / `MTLDepthStencilState` and
/// compute reflections, and holds them in `backend::metal::cache`.
///
/// No `m2v` field: AIR reaches Metal directly on this arm, so
/// `runtime::m2v_cache` is never populated and a zero there would read as an
/// empty cache rather than an absent rail.
pub(crate) fn emit_object_cache_levels() {
    let [functions, render_pso, compute_pso, samplers, depth_stencil, reflections] =
        super::cache::cache_levels();
    crate::observe::off(format!(
        "object_cache_levels (levels, not per-interval) functions={functions} \
         render_pso={render_pso} compute_pso={compute_pso} samplers={samplers} \
         depth_stencil={depth_stencil} reflections={reflections}"
    ));
}
