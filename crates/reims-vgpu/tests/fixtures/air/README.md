Small AIR fixture subset used by `reims-vgpu` Vulkan engine integration tests.

These files are not an in-tree metal2vulkan test suite. They are the minimal shader inputs needed
to keep product draw, compute, and reflection tests runnable against the public
`steelbrain/metal2vulkan` Git dependency.

`sample_texture_2d_nearest_clamp.{ll,air}` is the R9l compute-texture fixture: the same kernel body
the canonical rail's own C1b fixture carries (`metal-api-emulator`'s
`tests/fixtures/sample_texture_2d_nearest_clamp.ll`, owned synthetic), with the AIR target pair every
other fixture here carries added because this rail's translator reads the datalayout. The `.air` is
that text assembled by `llvm-as` (LLVM 22.1.8) and is never edited by hand: re-assemble from the
`.ll` when either changes. Tests carve it with `runtime::mtlb::extract_air`, exactly as a guest's
MTLB is carved.
