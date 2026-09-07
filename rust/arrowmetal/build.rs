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
    println!("cargo:rustc-env=ARROWMETAL_LIB_DIR={dir}");
}
