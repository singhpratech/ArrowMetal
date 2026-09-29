//! Refuses any target other than macOS on Apple silicon with a clear message, then repeats the
//! dylib's rpath for this crate's own tests and examples (docs/RUST.md, "Finding the dylib": a
//! build script's link arguments reach only the crate that owns it).

fn main() {
    println!("cargo:rerun-if-env-changed=ARROWMETAL_LIB");
    // docs.rs documents the crate without linking anything.
    if std::env::var_os("DOCS_RS").is_some() {
        return;
    }
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if os != "macos" || arch != "aarch64" {
        panic!(
            "datafusion-arrowmetal builds only for macOS on Apple silicon (aarch64-apple-darwin); \
             the target is {arch}-{os}. It runs DataFusion nodes on the GPU through \
             libArrowMetalC.dylib, a Metal library built for arm64 only."
        );
    }
    if let Ok(lib) = std::env::var("ARROWMETAL_LIB") {
        if let Some(dir) = std::path::Path::new(&lib).parent() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        }
    }
}
