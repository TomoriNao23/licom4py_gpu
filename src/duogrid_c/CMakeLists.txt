cmake_minimum_required(VERSION 3.10)
project(duogrid_coordinate VERSION 1.0 LANGUAGES Fortran C)
# Chtholly <mengleshan@mail.iap.ac.cn> 
# Created: 2025.9.15
# Updated: 2025.9.20 (set duogrid_coordinate the minimum_required)
# Updated: 2026.4.11 (remove MPI)

# Set Fortran compiler standard - use most permissive settings
set(CMAKE_Fortran_STANDARD 90)
set(CMAKE_Fortran_STANDARD_REQUIRED OFF)
set(CMAKE_Fortran_EXTENSIONS ON)

# Set C compiler standard - use most permissive settings
set(CMAKE_C_STANDARD 90)
set(CMAKE_C_STANDARD_REQUIRED OFF)
set(CMAKE_C_EXTENSIONS ON)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set output directories for all targets
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/modules)

# Create shared library
add_library(duogrid_coordinate SHARED)

# Collect source files
file(GLOB_RECURSE FMS_SOURCES 
  "chtholly/global_grid/*.F90"
  "chtholly/*.F90"
)

target_sources(duogrid_coordinate PRIVATE ${FMS_SOURCES})

set_target_properties(duogrid_coordinate PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
)

# GNU Fortran specific flags
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  target_compile_options(duogrid_coordinate PRIVATE 
    -fdefault-real-8
    -fdefault-double-8
    -ffree-form
    -ffree-line-length-none
    -fallow-invalid-boz
    -std=legacy
    -fallow-argument-mismatch
    -fno-range-check
    -fno-bounds-check
    -fcray-pointer
    -fno-implicit-none
    -fno-strict-overflow
    -w
  )
endif()