//! **`#[must_use]` on an obligation type says nothing about a `Vec` of them.**
//!
//! Several types here name something the caller must then do: destroy a
//! retired native object, tear down a replaced swapchain, release a claim.
//! Each carries `#[must_use]`, and the modules' own docs lean on that
//! annotation as the enforcement.
//!
//! `unused_must_use` does not look inside a `Vec`. A method returning
//! `Vec<Retired<T>>` whose result is dropped therefore compiles in silence,
//! and dropping it is exactly the leak the annotation on the element type was
//! written to prevent — the objects go without the device ever destroying
//! them, and nothing anywhere says so.
//!
//! So the annotation has to be on the method too, and whether it is there is a
//! fact about source text rather than about types. This is where that is held.

#[test]
fn every_method_handing_out_a_vec_of_obligations_says_so() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    // A syntactic instrument's one failure mode: matching nothing and passing
    // everywhere. This crate declares obligation types, so a scan that found
    // none of them found none of the methods either.
    let types = reims_vgpu_testkit::obligations::obligation_types_in(root);
    assert!(
        types.len() > 4,
        "the scan found {} obligation types in this crate, which is not what \
         it declares: {types:?}",
        types.len()
    );
    let found = reims_vgpu_testkit::obligations::unannotated_vec_returns(root);
    assert!(
        found.is_empty(),
        "these methods hand out obligations a caller may drop in silence, \
         because `#[must_use]` on the element type does not reach through a \
         `Vec`:\n{found:#?}"
    );
}
