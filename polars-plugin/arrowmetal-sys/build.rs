//! Finds libArrowMetalC.dylib and links it by rpath.
//!
//! Search order, first hit wins:
//!   1. `$ARROWMETAL_LIB_DIR`            -- a directory holding libArrowMetalC.dylib
//!   2. `$ARROWMETAL_LIB`                -- the dylib itself (the same variable the Python package reads)
//!   3. `<repo>/.build/release`, `<repo>/.build/debug`   -- a SwiftPM build in this checkout
//!   4. `<python package>/arrowmetal`    -- an installed wheel, if `ARROWMETAL_PYTHON_DIR` names one
//!   5. `/usr/local/lib`, `/opt/homebrew/lib`
//!
//! The dylib's install name is `@rpath/libArrowMetalC.dylib`, so the directory that was found is
//! also baked in as an `LC_RPATH` entry: the plugin then loads without `DYLD_LIBRARY_PATH`.

use std::path::{Path, PathBuf};

const LIB: &str = "libArrowMetalC.dylib";

fn repo_root() -> PathBuf {
    // <repo>/polars-plugin/arrowmetal-sys -> <repo>
    let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    here.parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or(here)
}

fn candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(dir) = std::env::var("ARROWMETAL_LIB_DIR") {
        out.push(PathBuf::from(dir));
    }
    if let Ok(lib) = std::env::var("ARROWMETAL_LIB") {
        if let Some(parent) = Path::new(&lib).parent() {
            out.push(parent.to_path_buf());
        }
    }
    let root = repo_root();
    out.push(root.join(".build/release"));
    out.push(root.join(".build/debug"));
    if let Ok(dir) = std::env::var("ARROWMETAL_PYTHON_DIR") {
        out.push(PathBuf::from(dir));
    }
    out.push(PathBuf::from("/usr/local/lib"));
    out.push(PathBuf::from("/opt/homebrew/lib"));
    out
}

fn main() {
    println!("cargo:rerun-if-env-changed=ARROWMETAL_LIB_DIR");
    println!("cargo:rerun-if-env-changed=ARROWMETAL_LIB");
    println!("cargo:rerun-if-env-changed=ARROWMETAL_PYTHON_DIR");

    let found = candidates().into_iter().find(|d| d.join(LIB).exists());
    let dir = match found {
        Some(d) => d,
        None => panic!(
            "{LIB} not found. Build it with\n    \
             DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
             swift build -c release --product ArrowMetalC\n\
             from the repository root, or point ARROWMETAL_LIB_DIR at the directory holding it."
        ),
    };
    let dir = dir.canonicalize().unwrap_or(dir);
    let dir = dir.display();

    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib=ArrowMetalC");
    // The install name is @rpath/libArrowMetalC.dylib, so the loader needs an rpath entry.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    // Downstream crates (the plugin cdylib) need the same rpath; cargo passes link args from a
    // build script only to the crate that owns it, so re-export the directory for them.
    println!("cargo:lib_dir={dir}");
    println!("cargo:rustc-env=ARROWMETAL_SYS_LIB_DIR={dir}");
}
