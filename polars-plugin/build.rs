//! The plugin is a `cdylib` that Polars `dlopen`s, so it needs its own `LC_RPATH` pointing at the
//! directory holding libArrowMetalC.dylib. Cargo passes `cargo:rustc-link-arg` from a build script
//! only to the crate that owns it, so `arrowmetal-sys` republishes the directory it found as
//! `DEP_ARROWMETALC_LIB_DIR` (through its `links = "ArrowMetalC"` key) and this picks it up.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if let Ok(dir) = std::env::var("DEP_ARROWMETALC_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }

    // pyo3 pulls in CPython symbols (`_PyUnicode_AsUTF8AndSize`, `__Py_IncRef`, ...). A Python
    // extension must not link libpython: the symbols are resolved from the interpreter that
    // dlopen()s it. maturin passes these two flags for us; a plain `cargo build` does not, and
    // without them the cdylib fails to link with "symbol(s) not found for architecture arm64".
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
