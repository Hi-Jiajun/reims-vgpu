//! Byte-level probe for `metal2vulkan` pin bumps: translate the fixture corpus
//! through the engine's own translation cache and write each SPIR-V module to
//! `PROBE_OUT_DIR`, so two pins can be compared module for module.
//!
//! The question a pin bump has to answer is "does the same AIR still leave the
//! engine as the same bytes", and only the engine's own path
//! (`runtime::m2v_cache`, the one the draw and compute rails call) can answer it
//! — the upstream CLI decides its own options and stage names. The dump is the
//! evidence; `sha256sum` over the two directories is the comparison.
//!
//! Inert unless `PROBE_OUT_DIR` is set, so it is a no-op in suite runs:
//!
//! ```text
//! PROBE_OUT_DIR=/tmp/m2v-bytes-before \
//!   cargo test -p reims-vgpu --no-default-features --features backend-vulkan \
//!   --test translation_bytes_probe -- --nocapture
//! ```

#![cfg(feature = "backend-vulkan")]

use std::path::PathBuf;

use metal2vulkan::passes::Stage;

/// Every fixture the engine's own tests translate, with the stage they are
/// translated at. A fixture may appear twice when two stages are exercised.
const FIXTURES: &[(&str, Stage)] = &[
    ("render_frag", Stage::Fragment),
    ("render_frag_color", Stage::Fragment),
    ("render_frag_fetch", Stage::Fragment),
    ("render_frag_lighting", Stage::Fragment),
    ("render_frag_static_sampler", Stage::Fragment),
    ("render_frag_texture", Stage::Fragment),
    ("render_frag_tint", Stage::Fragment),
    ("textured_quad", Stage::Fragment),
    ("textured_quad", Stage::Vertex),
    ("render_tri", Stage::Vertex),
    ("render_vtx_mesh", Stage::Vertex),
    ("reims_indexed_tri", Stage::Vertex),
    ("reims_indexed_tri_four_stream", Stage::Vertex),
    ("reims_indexed_tri_two_stream", Stage::Vertex),
    ("reims_indexed_tri_two_stream_precise", Stage::Vertex),
];

/// The kernel fixture, with the local size `vk_engine_compute.rs` runs it at.
const KERNELS: &[(&str, [u32; 3])] = &[("float_mul4_add3", [64, 1, 1])];

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/air")
}

fn stem(name: &str) -> String {
    name.replace('.', "_")
}

#[test]
fn dump_translation_bytes() {
    let Some(out) = std::env::var_os("PROBE_OUT_DIR") else {
        eprintln!("PROBE_OUT_DIR unset; skipping");
        return;
    };
    // Never share the live product logs with a concurrent boot.
    reims_vgpu::observe::redirect_logs_for_tests();
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).expect("create PROBE_OUT_DIR");

    for &(name, stage) in FIXTURES {
        let path = fixtures().join(format!("{name}.air"));
        let air = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        match reims_vgpu::runtime::m2v_cache::translate_cached_reflected(&air, stage, 1) {
            Ok(shader) => {
                let file = out.join(format!("{stage:?}__{}.spv", stem(name)));
                std::fs::write(&file, &shader.spirv).expect("write spirv");
                println!(
                    "probe name={name} stage={stage:?} bytes={} words={} file={}",
                    shader.spirv.len(),
                    shader.words.len(),
                    file.display()
                );
            }
            Err(error) => println!("probe name={name} stage={stage:?} declined={error}"),
        }
    }

    for &(name, local_size) in KERNELS {
        let path = fixtures().join(format!("{name}.air"));
        let air = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        match reims_vgpu::runtime::m2v_cache::translate_cached_kernel_reflected(&air, local_size, 1)
        {
            Ok(shader) => {
                let file = out.join(format!("Kernel__{}.spv", stem(name)));
                std::fs::write(&file, &shader.spirv).expect("write spirv");
                println!(
                    "probe name={name} stage=Kernel local_size={local_size:?} bytes={} words={} file={}",
                    shader.spirv.len(),
                    shader.words.len(),
                    file.display()
                );
            }
            Err(error) => println!("probe name={name} stage=Kernel declined={error}"),
        }
    }
}
