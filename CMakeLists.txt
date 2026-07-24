cmake_minimum_required(VERSION 3.20)
project(darcy_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -march=native is deliberately OFF by default: a wheel built with it crashes
# with SIGILL on any machine older than the build host. Turn it on for local
# benchmarking, never for a distributed artifact.
option(DARCY_NATIVE "Optimise for the build machine's CPU (-march=native)" OFF)
option(DARCY_BUILD_TESTS "Build the native C++ test binary" OFF)
option(DARCY_ENABLE_SANITIZERS "Build with address and UB sanitizers" OFF)

if(NOT DEFINED DARCY_VERSION)
  set(DARCY_VERSION "1.0.0")
endif()

# ---------------------------------------------------------------------------
# Core: header-only, so both the Python extension and the native tests compile
# the same code with the same flags.
# ---------------------------------------------------------------------------
add_library(darcy_core INTERFACE)
target_include_directories(darcy_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(darcy_core INTERFACE cxx_std_17)

# ---------------------------------------------------------------------------
# OpenMP. Apple's clang ships without the runtime, so FindOpenMP fails unless
# it is pointed at Homebrew's libomp. Probe for it, and degrade to a correct
# serial build rather than failing the install.
# ---------------------------------------------------------------------------
if(APPLE AND NOT DEFINED OpenMP_ROOT)
  find_program(DARCY_BREW brew)
  if(DARCY_BREW)
    execute_process(COMMAND ${DARCY_BREW} --prefix libomp
                    OUTPUT_VARIABLE DARCY_LIBOMP_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    if(DARCY_LIBOMP_PREFIX AND EXISTS "${DARCY_LIBOMP_PREFIX}/include/omp.h")
      set(OpenMP_ROOT "${DARCY_LIBOMP_PREFIX}")
      message(STATUS "darcy: using Homebrew libomp at ${OpenMP_ROOT}")
    endif()
  endif()
endif()

find_package(OpenMP QUIET COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
  target_link_libraries(darcy_core INTERFACE OpenMP::OpenMP_CXX)
  message(STATUS "darcy: OpenMP enabled")
else()
  message(WARNING
    "darcy: OpenMP not found -- building a correct but single-threaded solver. "
    "On macOS install it with 'brew install libomp'.")
endif()

if(MSVC)
  target_compile_options(darcy_core INTERFACE /O2 /W4 /permissive-)
else()
  target_compile_options(darcy_core INTERFACE
    -O3 -Wall -Wextra -Wpedantic -Wshadow -fno-math-errno)
  if(DARCY_NATIVE)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-march=native DARCY_HAS_MARCH_NATIVE)
    if(DARCY_HAS_MARCH_NATIVE)
      target_compile_options(darcy_core INTERFACE -march=native)
    endif()
  endif()
endif()

if(DARCY_ENABLE_SANITIZERS AND NOT MSVC)
  target_compile_options(darcy_core INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
  target_link_options(darcy_core INTERFACE -fsanitize=address,undefined)
endif()

# ---------------------------------------------------------------------------
# Python extension (skipped for a plain `cmake ..` developer build)
# ---------------------------------------------------------------------------
if(SKBUILD OR DARCY_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(_core MODULE src/bindings.cpp)
  target_link_libraries(_core PRIVATE darcy_core)
  target_compile_definitions(_core PRIVATE DARCY_VERSION="${DARCY_VERSION}")
  install(TARGETS _core DESTINATION darcy)
endif()

# ---------------------------------------------------------------------------
# Native C++ tests: numerical invariants that are cheaper to assert in C++
# than through the bindings (operator symmetry, Galerkin coarsening identity).
# ---------------------------------------------------------------------------
if(DARCY_BUILD_TESTS)
  enable_testing()
  add_executable(darcy_tests tests/cpp/test_core.cpp)
  target_link_libraries(darcy_tests PRIVATE darcy_core)
  add_test(NAME darcy_tests COMMAND darcy_tests)
endif()
