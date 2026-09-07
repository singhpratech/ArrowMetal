//! Finds `libArrowMetalC.dylib`, links it, and bakes its directory in as an `LC_RPATH` entry.
//!
//! The dylib's install name is `@rpath/libArrowMetalC.dylib`, so the directory that was found has to
//! be an rpath entry as well as a link-search path, or the test binary will not load.
//!
//! Search order, first hit wins:
//!   1. `$ARROWMETAL_LIB`      -- the full path to the dylib itself (the same variable the Python
//!                                package reads). This is the one to set.
//!   2. `$ARROWMETAL_LIB_DIR`  -- a directory holding it.
//!   3. `<repo>/.build/release`, then `<repo>/.build/debug` -- a SwiftPM build in this checkout.
//!   4. `/usr/local/lib`, `/opt/homebrew/lib` -- an installed copy.
//!
//! Anything else is a hard error with the message below; there is no fallback that would let the
//! crate build and then fail at run time with a dyld message.

use std::path::{Path, PathBuf};

const LIB: &str = "libArrowMetalC.dylib";

/// `<repo>/rust/arrowmetal-sys` -> `<repo>`.
fn repo_root() -> PathBuf {
    let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    here.parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or(here)
}

fn candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(lib) = std::env::var("ARROWMETAL_LIB") {
        if let Some(parent) = Path::new(&lib).parent() {
            out.push(parent.to_path_buf());
        }
    }
    if let Ok(dir) = std::env::var("ARROWMETAL_LIB_DIR") {
        out.push(PathBuf::from(dir));
    }
    let root = repo_root();
    out.push(root.join(".build/release"));
    out.push(root.join(".build/debug"));
    out.push(PathBuf::from("/usr/local/lib"));
    out.push(PathBuf::from("/opt/homebrew/lib"));
    out
}

fn main() {
    println!("cargo:rerun-if-env-changed=ARROWMETAL_LIB");
    println!("cargo:rerun-if-env-changed=ARROWMETAL_LIB_DIR");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        panic!(
            "arrowmetal-sys only builds on macOS: ArrowMetal is a Metal library and \
             {LIB} exists for Apple silicon only."
        );
    }

    let searched: Vec<String> = candidates().iter().map(|d| d.display().to_string()).collect();
    let dir = match candidates().into_iter().find(|d| d.join(LIB).exists()) {
        Some(d) => d,
        None => panic!(
            "{LIB} not found.\n\
             Searched, in order:\n  {}\n\n\
             Either point ARROWMETAL_LIB at the dylib:\n    \
             export ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib\n\
             or build it from the repository root:\n    \
             DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
             swift build -c release --product ArrowMetalC",
            searched.join("\n  ")
        ),
    };
    let dir = dir.canonicalize().unwrap_or(dir);
    println!("cargo:rerun-if-changed={}", dir.join(LIB).display());
    let dir = dir.display();

    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib=ArrowMetalC");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    // A build script's link args reach only the crate that owns it, so re-export the directory for
    // dependents (arrowmetal, and anything downstream of it) as DEP_ARROWMETALC_LIB_DIR.
    println!("cargo:lib_dir={dir}");
    println!("cargo:rustc-env=ARROWMETAL_SYS_LIB_DIR={dir}");
}
