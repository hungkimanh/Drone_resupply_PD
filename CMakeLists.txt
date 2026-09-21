cmake_minimum_required(VERSION 3.14)
project(DroneResupply CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type default: Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG   "/Zi /Od")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O2")
  set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -Wall -Wextra")
endif()

# OR-Tools (CP-SAT backend for LSP)
find_package(ortools REQUIRED HINTS /opt/homebrew/lib/cmake/ortools)
if(MINGW)
  # Official Windows OR-Tools archives are built for Visual Studio and export
  # MSVC-only interface flags. MinGW treats flags like /bigobj and -ignore:4221
  # as invalid inputs, so clear them from imported targets. Linking may still
  # require a compiler-compatible OR-Tools build.
  get_property(_imported_targets DIRECTORY PROPERTY IMPORTED_TARGETS)
  foreach(_target IN LISTS _imported_targets)
    set_property(TARGET ${_target} PROPERTY INTERFACE_COMPILE_OPTIONS "")
    set_property(TARGET ${_target} PROPERTY INTERFACE_LINK_OPTIONS "")
  endforeach()
endif()

# Shared library sources (no main)
set(PDP_SOURCES
    src/pdp_fitness.cpp
    src/pdp_ga.cpp
    src/pdp_init.cpp
    src/pdp_localsearch.cpp
    src/pdp_reader.cpp
    src/pdp_tabu.cpp
    src/pdp_utils.cpp
    src/pdp_validation.cpp
    # TS-ALNS-LSP architecture (new)
    src/pdp_lb.cpp
    src/pdp_lsp.cpp
    src/pdp_alns.cpp
    src/pdp_tabu_main.cpp
)

# --- Executable: pdp_solver ---
add_executable(pdp_solver
    src/main.cpp
    ${PDP_SOURCES}
)
target_include_directories(pdp_solver PRIVATE src)
target_link_libraries(pdp_solver PRIVATE ortools::ortools)

if(WIN32)
  get_filename_component(ORTOOLS_ROOT "${ortools_DIR}/../../.." ABSOLUTE)
  file(GLOB ORTOOLS_DLLS "${ORTOOLS_ROOT}/bin/*.dll")
  if(ORTOOLS_DLLS)
    add_custom_command(TARGET pdp_solver POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${ORTOOLS_DLLS}
              $<TARGET_FILE_DIR:pdp_solver>
      COMMENT "Copying OR-Tools runtime DLLs")
  endif()
endif()

# Output binary vào thư mục gốc
set_target_properties(pdp_solver
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

include(CTest)
if(BUILD_TESTING)
  add_executable(pdp_trip_boundary_test
      tests/pdp_trip_boundary_test.cpp
      src/pdp_fitness.cpp
  )
  target_include_directories(pdp_trip_boundary_test PRIVATE src)
  add_test(NAME pdp_trip_boundary_test COMMAND pdp_trip_boundary_test)
endif()

add_executable(pdp_decode_encoding
    src/decode_encoding_main.cpp
    src/pdp_fitness.cpp
    src/pdp_reader.cpp
    src/pdp_utils.cpp
    src/pdp_validation.cpp
)
target_include_directories(pdp_decode_encoding PRIVATE src)
