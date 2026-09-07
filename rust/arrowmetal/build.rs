//! `arrowmetal-sys` finds the dylib; this script only repeats its rpath.
//!
//! Cargo hands a build script's `rustc-link-arg` lines to the crate that owns the script and to
//! nothing downstream, so the `-rpath` `arrowmetal-sys` emits does not reach this crate's test and
//! example binaries. `arrowmetal-sys` re-exports the directory it found through its `links` key
//! (`DEP_ARROWMETALC_LIB_DIR`), and this repeats the flag here, which covers benches, binaries,
//! cdylibs, examples and tests.

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_ARROWMETALC_LIB_DIR");
    let dir = std::env::var("DEP_ARROWMETALC_LIB_DIR")
        .expect("arrowmetal-sys did not publish DEP_ARROWMETALC_LIB_DIR; its build script failed");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    // Deliberately not named ARROWMETAL_LIB_DIR: that is a user-facing *input* to the search in
    // arrowmetal-sys/build.rs, and a compile-time env of the same name reads confusingly in
    // `env!(..)` and shadows the input's meaning for anyone grepping. This is the resolved output.
    println!("cargo:rustc-env=ARROWMETAL_LINKED_LIB_DIR={dir}");
}
