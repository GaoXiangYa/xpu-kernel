# add_kernel_op(<name>)
#
# Builds the GoogleTest unit-test binary and (optionally) the benchmark binary
# for an operator. The operator directory follows this convention:
#
#   <name>/<name>.cuh       public interface, version table, CPU reference
#   <name>/<name>_v*.cu     one implementation per version (auto-discovered)
#   <name>/<name>_test.cpp  GoogleTest unit tests (required, host C++)
#   <name>/<name>_bench.cpp benchmark driver (optional, host C++)
#   <name>/CMakeLists.txt   add_kernel_op(<name>)
#
# Outputs:
#   <build>/bin/<name>_test   registered with CTest as `<name>`
#   <build>/bin/<name>_bench  run manually

# Applies the project-wide compile flags to one target.
function(_xpu_configure_target target)
  target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}" ${CUDAToolkit_INCLUDE_DIRS})
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
  )

  if(XPU_ENABLE_LINEINFO)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
  endif()

  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )
  endif()
endfunction()

function(add_kernel_op op_name)
  set(op_dir "${CMAKE_CURRENT_SOURCE_DIR}")

  file(GLOB op_version_sources CONFIGURE_DEPENDS "${op_dir}/${op_name}_v*.cu")
  if(NOT op_version_sources)
    message(FATAL_ERROR
      "add_kernel_op(${op_name}): no ${op_name}_v*.cu sources found in ${op_dir}")
  endif()
  list(SORT op_version_sources)

  set(op_test_source "${op_dir}/${op_name}_test.cpp")
  if(NOT EXISTS "${op_test_source}")
    message(FATAL_ERROR
      "add_kernel_op(${op_name}): missing GoogleTest driver ${op_test_source}")
  endif()

  add_executable(${op_name}_test ${op_test_source} ${op_version_sources})
  target_link_libraries(${op_name}_test PRIVATE xpu_common GTest::gtest_main CUDA::cudart)
  _xpu_configure_target(${op_name}_test)

  add_test(NAME ${op_name} COMMAND ${op_name}_test)
  set_tests_properties(${op_name} PROPERTIES TIMEOUT 300)

  list(LENGTH op_version_sources num_versions)
  message(STATUS "  + ${op_name}: ${num_versions} version(s) -> bin/${op_name}_test")

  set(op_bench_source "${op_dir}/${op_name}_bench.cpp")
  if(EXISTS "${op_bench_source}")
    add_executable(${op_name}_bench ${op_bench_source} ${op_version_sources})
    target_link_libraries(${op_name}_bench PRIVATE xpu_common CUDA::cudart)
    _xpu_configure_target(${op_name}_bench)
    message(STATUS "    + ${op_name} benchmark -> bin/${op_name}_bench")
  endif()
endfunction()
