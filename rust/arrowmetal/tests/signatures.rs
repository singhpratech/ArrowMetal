//! Checks every `arrowmetal-sys` declaration against `include/arrowmetal.h`.
//!
//! The `-sys` crate is hand-written, so the claim "every entry point the safe crate uses is declared
//! with the exact C signature" needs something behind it. This test parses both files at test time
//! and compares, for every `am_*` function the crate declares: that the header has it at all, that
//! the arity matches, and that each parameter and the return value agree on base type and pointer
//! depth.
//!
//! Constness and mutability are deliberately not compared -- `const char**` and `*mut *const c_char`
//! are the same ABI -- but a wrong width, a wrong arity, a missing function or a swapped type is a
//! failure here rather than undefined behaviour at run time.

use std::collections::BTreeMap;
use std::path::PathBuf;

/// A parameter or return type reduced to what the ABI actually cares about.
#[derive(Debug, PartialEq, Eq, Clone)]
struct Ty {
    base: String,
    stars: usize,
}

impl std::fmt::Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.base, "*".repeat(self.stars))
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Sig {
    ret: Ty,
    params: Vec<Ty>,
}

fn repo_root() -> PathBuf {
    // <repo>/rust/arrowmetal -> <repo>
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

/// Drops `/* ... */` and `// ...`, which both appear inside the header's parameter lists.
fn strip_c_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let b = src.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(b.len());
            out.push(' ');
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

/// `const char* name` -> `char*`; `am_array** out` -> `am_array**`; `void` -> nothing.
fn parse_c_type(decl: &str) -> Option<Ty> {
    let decl = decl.replace('\n', " ");
    let stars = decl.matches('*').count();
    let words: Vec<&str> = decl
        .split(|c: char| c == '*' || c.is_whitespace())
        .filter(|w| !w.is_empty() && !matches!(*w, "const" | "struct" | "unsigned" | "signed"))
        .collect();
    if words.is_empty() {
        return None;
    }
    // The last word is the parameter name unless there is only one word (a bare type, as in a
    // return type or an unnamed parameter).
    let base = if words.len() == 1 { words[0] } else { words[words.len() - 2] };
    if base == "void" && stars == 0 {
        return None;
    }
    Some(Ty { base: base.to_string(), stars })
}

/// Splits a parameter list on commas that are not inside parentheses.
fn split_params(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' => {
                depth += 1;
                cur.push(c);
            }
            ')' => {
                depth = depth.saturating_sub(1);
                cur.push(c);
            }
            ',' if depth == 0 => {
                out.push(std::mem::take(&mut cur));
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

/// Every `am_*` function declaration in `include/arrowmetal.h`.
fn parse_header() -> BTreeMap<String, Sig> {
    let path = repo_root().join("include/arrowmetal.h");
    let src = strip_c_comments(
        &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    );
    let mut out = BTreeMap::new();
    for stmt in src.split(';') {
        let stmt = stmt.trim();
        if stmt.starts_with("typedef") || stmt.starts_with('#') {
            continue;
        }
        let Some(open) = stmt.find('(') else { continue };
        let Some(close) = stmt.rfind(')') else { continue };
        if close < open {
            continue;
        }
        let head = &stmt[..open];
        // The name is the last identifier before the '('.
        let Some(name_start) = head.rfind(|c: char| !(c.is_alphanumeric() || c == '_')) else {
            continue;
        };
        let name = head[name_start + 1..].trim();
        if !name.starts_with("am_") {
            continue;
        }
        let ret_decl = head[..name_start + 1].trim();
        let Some(ret) = parse_c_type(&format!("{ret_decl} r")).or_else(|| {
            // A `void` return parses to None; represent it as the unit type.
            Some(Ty { base: "void".into(), stars: 0 })
        }) else {
            continue;
        };
        let params: Vec<Ty> =
            split_params(&stmt[open + 1..close]).iter().filter_map(|p| parse_c_type(p)).collect();
        out.insert(name.to_string(), Sig { ret, params });
    }
    out
}

/// Maps a Rust FFI type onto the C type it stands for.
fn parse_rust_type(decl: &str) -> Option<Ty> {
    let decl = decl.trim();
    if decl.is_empty() {
        return None;
    }
    let stars = decl.matches("*mut ").count() + decl.matches("*const ").count();
    let base = decl
        .replace("*mut ", "")
        .replace("*const ", "")
        .replace("sys::", "")
        .replace("super::", "")
        .trim()
        .to_string();
    let base = match base.as_str() {
        "c_int" => "int",
        "c_char" => "char",
        "c_void" => "void",
        "i8" => "int8_t",
        "u8" => "uint8_t",
        "i16" => "int16_t",
        "u16" => "uint16_t",
        "i32" => "int32_t",
        "u32" => "uint32_t",
        "i64" => "int64_t",
        "u64" => "uint64_t",
        "f32" => "float",
        "f64" => "double",
        other => other,
    };
    Some(Ty { base: base.to_string(), stars })
}

/// Every `pub fn am_*` inside the `unsafe extern "C"` block of `arrowmetal-sys`.
fn parse_sys_crate() -> BTreeMap<String, Sig> {
    let path = repo_root().join("rust/arrowmetal-sys/src/lib.rs");
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    // Drop doc and line comments so a `//` example cannot be mistaken for a declaration.
    let src: String = src
        .lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n");

    let mut out = BTreeMap::new();
    let mut rest = src.as_str();
    while let Some(i) = rest.find("pub fn am_") {
        rest = &rest[i + "pub fn ".len()..];
        let Some(open) = rest.find('(') else { break };
        let name = rest[..open].trim().to_string();
        // Find the matching ')'.
        let mut depth = 0usize;
        let mut close = None;
        for (j, c) in rest[open..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(open + j);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(close) = close else { break };
        let params: Vec<Ty> = split_params(&rest[open + 1..close])
            .iter()
            .filter_map(|p| {
                let p = p.trim();
                if p.is_empty() {
                    return None;
                }
                // `name: type`
                let ty = p.split_once(':').map(|(_, t)| t).unwrap_or(p);
                parse_rust_type(ty)
            })
            .collect();
        // The return type runs from the ')' to the ';'.
        let tail_end = rest[close..].find(';').map(|k| close + k).unwrap_or(rest.len());
        let tail = rest[close + 1..tail_end].trim();
        let ret = match tail.strip_prefix("->") {
            Some(t) => parse_rust_type(t).unwrap_or(Ty { base: "void".into(), stars: 0 }),
            None => Ty { base: "void".into(), stars: 0 },
        };
        out.insert(name, Sig { ret, params });
        rest = &rest[tail_end..];
    }
    out
}

/// The parser has to actually find things, or the test would pass by finding nothing to compare.
#[test]
fn the_parsers_find_declarations() {
    let header = parse_header();
    let sys = parse_sys_crate();
    assert!(header.len() > 200, "parsed only {} declarations from the header", header.len());
    assert!(sys.len() > 30, "parsed only {} declarations from arrowmetal-sys", sys.len());
    assert!(header.contains_key("am_reduce"), "the header parser missed am_reduce");
    assert!(sys.contains_key("am_reduce"), "the sys parser missed am_reduce");
}

/// Every entry point the crate declares must exist in the header with the same signature.
#[test]
fn every_sys_declaration_matches_the_header() {
    let header = parse_header();
    let sys = parse_sys_crate();

    let mut problems = Vec::new();
    for (name, mine) in &sys {
        let Some(theirs) = header.get(name) else {
            problems.push(format!("{name}: declared in arrowmetal-sys but not in the header"));
            continue;
        };
        if mine.params.len() != theirs.params.len() {
            problems.push(format!(
                "{name}: {} parameters declared, header has {}",
                mine.params.len(),
                theirs.params.len()
            ));
            continue;
        }
        if mine.ret != theirs.ret {
            problems.push(format!(
                "{name}: returns {} , header says {}",
                mine.ret, theirs.ret
            ));
        }
        for (i, (a, b)) in mine.params.iter().zip(&theirs.params).enumerate() {
            if a != b {
                problems.push(format!("{name}: parameter {i} is {a}, header says {b}"));
            }
        }
    }
    assert!(problems.is_empty(), "arrowmetal-sys has drifted from the header:\n  {}", problems.join("\n  "));
}
