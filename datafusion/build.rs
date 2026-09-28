//! Repeats the dylib's rpath for this crate's own tests and examples (docs/RUST.md, "Finding the
//! dylib": a build script's link arguments reach only the crate that owns it).

fn main() {
    println!("cargo:rerun-if-env-changed=ARROWMETAL_LIB");
    if let Ok(lib) = std::env::var("ARROWMETAL_LIB") {
        if let Some(dir) = std::path::Path::new(&lib).parent() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        }
    }
}
