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

`render_frag_buffer_read_and_unused.{ll,air}` is the R9n mixed-slot fixture: one fragment entry with
two `[[buffer(N)]]` arguments, `[[buffer(0)]]` read into the red channel and `[[buffer(1)]]` declared
by the metadata and never dereferenced. It is the shape the stage-buffer door's unread population
actually has — several declared buffers, only some of them reached — and the one
`provider_render_rail.rs` pins the drop against: the reached slot is still stated with its view
while the unread one takes no part in the canonical pairing.

`render_frag_fetch_texture_2d{,_far}.{ll,air}` and `render_frag_fetch_and_sample.{ll,air}` are the
R15 texel-fetch fixtures: fragment entries whose `[[texture(0)]]` argument is written
`texture2d<float, read>` in Metal's type system, so the translator lowers `texture.read()` to
`OpImageFetch` and no `OpSampledImage` — the shape the census v11 read as "sample sites name no
runtime `[[sampler(n)]]`". The two `_2d` siblings differ only in the texels they read, which is what
makes "the frame followed the coordinates" falsifiable; `render_frag_fetch_and_sample` carries a
second, sampled `[[texture(1)]]` beside a runtime `[[sampler(0)]]` argument, which is the shape the
guest's own captures have (one fetched texture beside runtime-sampled ones). The `.ll` sources are
the same bodies as `metal-api-emulator`'s owned synthetic `render_fetch_texture_2d(.far).frag.ll`
fixtures (E-RS4), with this corpus's datalayout pair — the one this rail's translator reads — and
this rail's entry names; re-assemble them from the `.ll` with `llvm-as` 22.1.8 like every other
fixture here.

`render_frag_runtime_sampler_index3.{ll,air}` is the R16 sparse-declaration fixture: the
runtime-sampler fixture's own body with its one texture argument moved to `[[texture(3)]]`, so the
stage's texture argument table has nothing at 0, 1 or 2. It is the shape the census v12 read 35269
times (57.8% of that boot's first-failure lines, every one of them the seam's positional refusal)
and the shape the captured modules behind those lines carry: a texture declared at Metal index 3
beside a runtime `[[sampler(0)]]` argument. The `.ll` is the same body as
`render_frag_runtime_sampler.ll` (R12) under the new index, which is what makes "the index is a
binding and not a position" a byte-for-byte comparison against that fixture's frame; re-assemble it
from the `.ll` with `llvm-as` 22.1.8 like every other fixture here.
