#!/usr/bin/env python3
"""Turn a plain shared library into a loadable `.duckdb_extension`.

DuckDB will not dlopen an arbitrary dylib. It first reads a 512-byte footer that says which platform
the binary was built for, which ABI it speaks, and which version it was built against, and refuses
the file if any of that disagrees with the running engine. This script writes that footer.

The layout is DuckDB's, and is the same one `scripts/append_extension_metadata.py` in
duckdb/extension-ci-tools writes:

    <the shared library, unchanged>
    <a 20-byte WebAssembly custom-section header, so the same footer is valid in a Wasm build>
    field 8   32 bytes   unused
    field 7   32 bytes   unused
    field 6   32 bytes   unused
    field 5   32 bytes   ABI type          "C_STRUCT" for a C-API extension
    field 4   32 bytes   extension version
    field 3   32 bytes   DuckDB version    -- for C_STRUCT this is the *C API* version, e.g. v1.2.0
    field 2   32 bytes   platform          e.g. osx_arm64
    field 1   32 bytes   the literal "4"   the marker that identifies a DuckDB extension at all
    signature 256 bytes  zero, i.e. unsigned

Every field is ASCII, NUL-padded to 32 bytes, and the fields are written high number first. The
output of this script has been diffed against that of DuckDB's own script for the same inputs and is
byte-identical, so an extension it stamps is indistinguishable from one the DuckDB CI produced.

An unsigned extension loads only into a database started with `allow_unsigned_extensions`; see
docs/DUCKDB.md.

    python3 append_metadata.py -l build/libarrowmetal_extension.dylib \
        -o build/arrowmetal.duckdb_extension -p osx_arm64 -dv v1.2.0 -ev 0.1.0
"""
import argparse
import shutil


def wasm_custom_section_header():
    """The 20 bytes that make the footer a valid WebAssembly custom section named 'duckdb_signature'.

    Native builds ignore this; it is here so a native and a Wasm extension carry byte-identical
    metadata, which is what DuckDB's own tooling produces."""
    out = bytes([0])                       # section id 0: custom section
    out += bytes([147, 4])                 # LEB128 length of the rest: 1 + 16 + 2 + 8*32 + 256 = 531
    out += bytes([16])                     # length of the section name
    out += b"duckdb_signature"
    out += bytes([128, 4])                 # LEB128 512: the payload that follows
    return out


def field(text):
    encoded = text.encode("ascii")
    if len(encoded) > 32:
        raise SystemExit(f"metadata field {text!r} is longer than 32 bytes")
    return encoded + b"\x00" * (32 - len(encoded))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-l", "--library-file", required=True, help="the shared library to wrap")
    parser.add_argument("-o", "--out-file", required=True, help="the .duckdb_extension to write")
    parser.add_argument("-p", "--duckdb-platform", required=True, help="e.g. osx_arm64")
    parser.add_argument("-dv", "--duckdb-version", required=True,
                        help="the C API version for a C_STRUCT extension, e.g. v1.2.0")
    parser.add_argument("-ev", "--extension-version", required=True)
    parser.add_argument("--abi-type", default="C_STRUCT")
    args = parser.parse_args()

    shutil.copyfile(args.library_file, args.out_file)
    with open(args.out_file, "ab") as handle:
        handle.write(wasm_custom_section_header())
        handle.write(field(""))                          # field 8, unused
        handle.write(field(""))                          # field 7, unused
        handle.write(field(""))                          # field 6, unused
        handle.write(field(args.abi_type))               # field 5
        handle.write(field(args.extension_version))      # field 4
        handle.write(field(args.duckdb_version))         # field 3
        handle.write(field(args.duckdb_platform))        # field 2
        handle.write(field("4"))                         # field 1, the DuckDB extension marker
        handle.write(b"\x00" * 256)                      # unsigned

    print(f"wrote {args.out_file}: {args.abi_type} extension {args.extension_version} "
          f"for {args.duckdb_platform}, C API {args.duckdb_version}")


if __name__ == "__main__":
    main()
