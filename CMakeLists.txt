cmake_minimum_required(VERSION 3.24)

project(xpu-kernel VERSION 0.1.0 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# "native" detects the local GPU. On a machine without a visible GPU (CI,
# remote build host), configure with e.g. -DCMAKE_CUDA_ARCHITECTURES=86.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING
      "CUDA architectures, e.g. '86', '89', '80;90', or 'native'")
endif()

option(XPU_ENABLE_LINEINFO "Add -lineinfo to CUDA builds (useful for ncu/Nsight)" ON)
option(XPU_BUILD_TESTS "Build the test/benchmark executable of every operator" ON)

if(XPU_BUILD_TESTS)
  enable_testing()

  # GoogleTest: use the system package when available, otherwise fetch a
  # pinned release. Tests link GTest::gtest_main.
  find_package(GTest QUIET)
  if(GTest_FOUND)
    message(STATUS "GoogleTest: system package")
  else()
    message(STATUS "GoogleTest: not found locally, fetching googletest v1.15.2")
    include(FetchContent)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.15.2
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(googletest)
  endif()
endif()

include(cmake/KernelOp.cmake)

add_subdirectory(common)

# ---------------------------------------------------------------------------
# Operators
#
# Every top-level directory that contains both `<dir>/CMakeLists.txt` and
# `<dir>/<dir>.cuh` is treated as one operator. Add a new folder and reconfigure
# (CMake re-runs automatically thanks to CONFIGURE_DEPENDS).
# ---------------------------------------------------------------------------
if(XPU_BUILD_TESTS)
  set(XPU_OPERATORS "")
  file(GLOB XPU_OPERATOR_DIRS CONFIGURE_DEPENDS LIST_DIRECTORIES true
       "${CMAKE_CURRENT_SOURCE_DIR}/*")

  foreach(op_dir IN LISTS XPU_OPERATOR_DIRS)
    if(NOT IS_DIRECTORY "${op_dir}")
      continue()
    endif()

    get_filename_component(op_name "${op_dir}" NAME)
    if(EXISTS "${op_dir}/CMakeLists.txt" AND EXISTS "${op_dir}/${op_name}.cuh")
      add_subdirectory("${op_dir}" "${CMAKE_BINARY_DIR}/operators/${op_name}")
      list(APPEND XPU_OPERATORS "${op_name}")
    endif()
  endforeach()

  message(STATUS "xpu-kernel operators: ${XPU_OPERATORS}")
endif()
