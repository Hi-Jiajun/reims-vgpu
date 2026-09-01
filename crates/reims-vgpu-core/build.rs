//! Tells the fixture test whether Apple's captured records are on disk.
//!
//! The same probe the wire crate and the device crate use, included rather than
//! copied for the reason `fixture_presence.rs` gives: the directory override,
//! the file names and what `REIMS_WIRE_FIXTURES_REQUIRED` means must not fork
//! between the suites that stand down without them.

include!("../reims-vgpu-wire/oracle/fixture_presence.rs");

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../reims-vgpu-wire/oracle/fixture_presence.rs");
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    probe_wire_fixtures(&format!("{manifest}/../reims-vgpu-wire/fixtures"));
}
