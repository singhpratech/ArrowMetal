# ArrowMetal's DuckDB extension.
#
# This is the same compile build.sh performs, for people who would rather drive it from cmake.
# Neither path needs a DuckDB source tree: the extension speaks DuckDB's C extension API, so all it
# needs is duckdb.h and duckdb_extension.h (build.sh downloads them into third_party/, or drop them
# there yourself from the libduckdb release archive).
#
#   cmake -S duckdb-extension -B duckdb-extension/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build duckdb-extension/build
#
# The result is duckdb-extension/build/arrowmetal.duckdb_extension, ready for LOAD.
cmake_minimum_required(VERSION 3.16)
project(arrowmetal_duckdb_extension LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

get_filename_component(ARROWMETAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# The extension's version, and the DuckDB *C API* version it declares. A C_STRUCT extension built
# against v1.2.0 loads into any DuckDB whose C API is at least that; the engine's own version does
# not have to match. See docs/DUCKDB.md.
set(EXTENSION_VERSION "0.1.0" CACHE STRING "version stamped into the extension metadata")
set(DUCKDB_C_API_VERSION "v1.2.0" CACHE STRING "DuckDB C extension API version to declare")

# The platform string DuckDB checks before it will dlopen the file.
if(NOT DEFINED DUCKDB_PLATFORM)
  if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(DUCKDB_PLATFORM "osx_arm64")
  elseif(APPLE)
    set(DUCKDB_PLATFORM "osx_amd64")
  else()
    message(FATAL_ERROR "set -DDUCKDB_PLATFORM=... for this system")
  endif()
endif()

set(ARROWMETAL_LIB "${ARROWMETAL_ROOT}/.build/release/libArrowMetalC.dylib"
    CACHE FILEPATH "libArrowMetalC.dylib to link against")
if(NOT EXISTS "${ARROWMETAL_LIB}")
  message(FATAL_ERROR
    "libArrowMetalC.dylib not found at ${ARROWMETAL_LIB}.\n"
    "Build it first: DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer "
    "swift build -c release --product ArrowMetalC")
endif()

foreach(header duckdb.h duckdb_extension.h)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/${header}")
    message(FATAL_ERROR
      "${header} not found in duckdb-extension/third_party/.\n"
      "Run duckdb-extension/build.sh once to fetch the DuckDB C headers, or copy them there from "
      "the libduckdb release archive.")
  endif()
endforeach()

add_library(arrowmetal_extension SHARED src/arrowmetal_extension.cpp)
target_include_directories(arrowmetal_extension PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
  "${ARROWMETAL_ROOT}/include")
target_link_libraries(arrowmetal_extension PRIVATE "${ARROWMETAL_LIB}")
get_filename_component(ARROWMETAL_LIB_DIR "${ARROWMETAL_LIB}" DIRECTORY)
set_target_properties(arrowmetal_extension PROPERTIES
  BUILD_RPATH "${ARROWMETAL_LIB_DIR}"
  INSTALL_RPATH "${ARROWMETAL_LIB_DIR}")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# DuckDB refuses to load a bare shared library: it wants a 512-byte metadata footer naming the
# platform, the ABI and the version. append_metadata.py writes it.
add_custom_command(TARGET arrowmetal_extension POST_BUILD
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/append_metadata.py"
          -l "$<TARGET_FILE:arrowmetal_extension>"
          -o "${CMAKE_BINARY_DIR}/arrowmetal.duckdb_extension"
          -p "${DUCKDB_PLATFORM}"
          -dv "${DUCKDB_C_API_VERSION}"
          -ev "${EXTENSION_VERSION}"
  COMMENT "Appending DuckDB extension metadata")
