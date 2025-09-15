cmake_minimum_required(VERSION 3.10)
project(FMS_Chtholly VERSION 1.0 LANGUAGES Fortran C)
# Chtholly <mengleshan@mail.iap.ac.cn> 
# Created: 2025.9.15
# Updated: 2025.9.15


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
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/lib)

# Set Fortran module output directory
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/modules)

# Compiler-specific options - add more compatibility options
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fdefault-real-8 -fdefault-double-8")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ffree-form -ffree-line-length-none")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fPIC -fallow-invalid-boz")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=legacy -fallow-argument-mismatch")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fno-range-check -fno-bounds-check")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fcray-pointer -fallow-argument-mismatch")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fno-implicit-none -fno-strict-overflow")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -w")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -r8 -double-size 128")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -free -fPIC")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std-legacy -no-range-check")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -assume byterecl -assume old_ldout_format")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -w")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Cray")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -s real64 -s real128")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -f free -fPIC")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -h noomp -h noacc")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -w")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "PGI")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -r8 -i8 -Mfree -Mextend")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fPIC -Ktrap=fp")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -w")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "NAG")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -double -free -PIC")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -f2003 -f2008 -f95")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -w")
else()
  # TODO 通用选项，适用于其他编译器
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fPIC")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -w")
endif()

# Add C compiler flags to suppress warnings
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
elseif(CMAKE_C_COMPILER_ID STREQUAL "Intel")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
elseif(CMAKE_C_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
endif()

# Check MPI - use more permissive checking
find_package(MPI)
if(MPI_FOUND)
  message(STATUS "MPI found: ${MPI_C_COMPILER}")
  message(STATUS "MPI Fortran compiler: ${MPI_Fortran_COMPILER}")
  if(MPI_C_COMPILER)
    set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
  endif()
  if(MPI_Fortran_COMPILER)
    set(CMAKE_Fortran_COMPILER ${MPI_Fortran_COMPILER})
  endif()
else()
  message(WARNING "MPI not found, proceeding without MPI support")
  set(MPI_FOUND FALSE)
endif()

# Check netCDF - use more permissive checking
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(NETCDF netcdf)
  pkg_check_modules(NETCDFF netcdf-fortran)
endif()

# If pkg-config fails, try other methods
if(NOT NETCDF_FOUND)
  find_library(NETCDF_LIBRARIES NAMES netcdf)
  find_path(NETCDF_INCLUDE_DIRS NAMES netcdf.h)
  if(NETCDF_LIBRARIES AND NETCDF_INCLUDE_DIRS)
    set(NETCDF_FOUND TRUE)
  endif()
endif()

if(NOT NETCDFF_FOUND)
  find_library(NETCDFF_LIBRARIES NAMES netcdff)
  find_path(NETCDFF_INCLUDE_DIRS NAMES netcdf.inc)
  if(NETCDFF_LIBRARIES AND NETCDFF_INCLUDE_DIRS)
    set(NETCDFF_FOUND TRUE)
  endif()
endif()

# If still can't find netCDF, set default values
if(NOT NETCDF_FOUND)
  message(WARNING "netCDF C library not found, proceeding without netCDF support")
  set(NETCDF_FOUND FALSE)
  set(NETCDF_LIBRARIES "")
  set(NETCDF_INCLUDE_DIRS "")
endif()

if(NOT NETCDFF_FOUND)
  message(WARNING "netCDF Fortran library not found, proceeding without netCDF Fortran support")
  set(NETCDFF_FOUND FALSE)
  set(NETCDFF_LIBRARIES "")
  set(NETCDFF_INCLUDE_DIRS "")
endif()

# Force set netCDF library paths (based on pkg-config output)
if(PKG_CONFIG_FOUND)
  set(NETCDF_LIBRARY_DIRS "/opt/homebrew/Cellar/netcdf-fortran/4.6.2/lib;/opt/homebrew/Cellar/netcdf/4.9.3/lib")
  set(NETCDF_LIBRARIES "netcdff;netcdf")
  set(NETCDFF_LIBRARIES "netcdff")
  set(NETCDF_FOUND TRUE)
  set(NETCDFF_FOUND TRUE)
  message(STATUS "Using hardcoded netCDF paths for macOS")
endif()

if(NETCDF_FOUND)
  message(STATUS "netCDF C: ${NETCDF_LIBRARIES}")
endif()
if(NETCDFF_FOUND)
  message(STATUS "netCDF Fortran: ${NETCDFF_LIBRARIES}")
endif()

# Set include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/mpp/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/fms)
if(NETCDF_FOUND)
  include_directories(${NETCDF_INCLUDE_DIRS})
endif()
if(NETCDFF_FOUND)
  include_directories(${NETCDFF_INCLUDE_DIRS})
endif()

# Set link directories
if(NETCDF_FOUND)
  link_directories(${NETCDF_LIBRARY_DIRS})
endif()
if(NETCDFF_FOUND)
  link_directories(${NETCDFF_LIBRARY_DIRS})
endif()

# Define preprocessor macros - use conditional definitions
if(MPI_FOUND)
  add_definitions(-Duse_libMPI)
endif()
if(NETCDF_FOUND)
  add_definitions(-Duse_netCDF)
endif()

# Detect operating system
if(APPLE)
  add_definitions(-D__APPLE__)
endif()

# Detect CRAY system
if(DEFINED ENV{CRAY_PRGENVINTEL} OR DEFINED ENV{CRAY_PRGENVGNU} OR DEFINED ENV{CRAY_PRGENVCRAY})
  add_definitions(-DCRAY)
endif()

# Fortran module output directory is already set above

# Create unified FMS library
add_library(fms_unified SHARED)

# Use simpler method - collect all source files
file(GLOB_RECURSE FMS_SOURCES 
  "constants/*.F90"
  "platform/*.F90"
  "mpp/*.F90"
  "mpp/*.c"
  "memutils/*.F90"
  "memutils/*.c"
  "fms/*.F90"
  "libFMS/*.F90"
  "diag_manager/*.F90"
  "field_manager/*.F90"
  "global_grid/*.F90"
  "mosaic/*.F90"
  "mosaic/*.c"
  "mosaic/*.h"
  "time_manager/*.F90"
  "horiz_interp/*.F90"
  "interpolator/*.F90"
  "tracer_manager/*.F90"
  "topography/*.F90"
  "sat_vapor_pres/*.F90"
  "random_numbers/*.F90"
  "oda_tools/*.F90"
  "oda_tools/*.f90"
  "station_data/*.F90"
  "time_interp/*.F90"
  "tridiagonal/*.F90"
  "monin_obukhov/*.F90"
  "column_diagnostics/*.F90"
  "data_override/*.F90"
  "coupler/*.F90"
  "exchange/*.F90"
  "fft/*.F90"
  "drifters/*.F90"
  "astronomy/*.F90"
  "axis_utils/*.F90"
  "block_control/*.F90"
  "amip_interp/*.F90"
)

# Add chtholly source files after mpp (for dependency order)
file(GLOB CHTHOLLY_SOURCES "chtholly/*.F90")

# Merge all source files
set(FMS_ALL_SOURCES ${FMS_SOURCES} ${CHTHOLLY_SOURCES})

# Add C source files
file(GLOB_RECURSE FMS_C_SOURCES
  "mpp/*.c"
  "memutils/*.c"
  "mosaic/*.c"
  "mosaic/*.h"
)

# Merge all source files
set(FMS_ALL_SOURCES ${FMS_SOURCES} ${CHTHOLLY_SOURCES} ${FMS_C_SOURCES})

# Add source files to unified library
target_sources(fms_unified PRIVATE ${FMS_ALL_SOURCES})

# Set compilation order - ensure dependency modules compile first
set_source_files_properties(
  "fft/fft99.F90"
  PROPERTIES
  OBJECT_OUTPUTS "${CMAKE_CURRENT_BINARY_DIR}/fft/fft99.F90.o"
)

set_source_files_properties(
  "fft/fft.F90"
  PROPERTIES
  OBJECT_DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/fft/fft99.F90.o"
)

# Ensure chtholly modules compile after mpp modules
# Note: CMake will automatically handle module dependencies

# Set target properties
set_target_properties(fms_unified PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
)

# Set include directories
target_include_directories(fms_unified PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/mpp/include
  ${CMAKE_SOURCE_DIR}/fms
)
if(NETCDF_FOUND)
  target_include_directories(fms_unified PUBLIC ${NETCDF_INCLUDE_DIRS})
endif()
if(NETCDFF_FOUND)
  target_include_directories(fms_unified PUBLIC ${NETCDFF_INCLUDE_DIRS})
endif()

# Link libraries - use conditional linking
if(MPI_FOUND)
  target_link_libraries(fms_unified PUBLIC MPI::MPI_C MPI::MPI_Fortran)
endif()

# Link netCDF libraries
if(NETCDF_FOUND)
  target_link_libraries(fms_unified PUBLIC ${NETCDF_LIBRARIES})
  message(STATUS "Linking netCDF C: ${NETCDF_LIBRARIES}")
endif()
if(NETCDFF_FOUND)
  target_link_libraries(fms_unified PUBLIC ${NETCDFF_LIBRARIES})
  message(STATUS "Linking netCDF Fortran: ${NETCDFF_LIBRARIES}")
endif()

# If netCDF found by pkg-config, use its library paths
if(PKG_CONFIG_FOUND AND NETCDF_FOUND)
  target_link_directories(fms_unified PUBLIC ${NETCDF_LIBRARY_DIRS})
endif()
if(PKG_CONFIG_FOUND AND NETCDFF_FOUND)
  target_link_directories(fms_unified PUBLIC ${NETCDFF_LIBRARY_DIRS})
endif()

# Set Fortran compiler options
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  target_compile_options(fms_unified PRIVATE 
    $<$<COMPILE_LANGUAGE:Fortran>:-fdefault-real-8 -std=legacy -fallow-argument-mismatch -fcray-pointer -fno-implicit-none>
  )
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  target_compile_options(fms_unified PRIVATE 
    $<$<COMPILE_LANGUAGE:Fortran>:-r8 -std-legacy -no-range-check>
  )
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Cray")
  target_compile_options(fms_unified PRIVATE 
    $<$<COMPILE_LANGUAGE:Fortran>:-s real64 -h noomp -h noacc>
  )
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "PGI")
  target_compile_options(fms_unified PRIVATE 
    $<$<COMPILE_LANGUAGE:Fortran>:-r8 -i8 -Mfree -Mextend>
  )
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "NAG")
  target_compile_options(fms_unified PRIVATE 
    $<$<COMPILE_LANGUAGE:Fortran>:-double -free -f95>
  )
endif()

# Set C compiler options
if(CMAKE_C_COMPILER_ID MATCHES "GNU")
  target_compile_options(fms_unified PRIVATE 
    $<$<COMPILE_LANGUAGE:C>:-fPIC>
  )
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
  target_compile_options(fms_unified PRIVATE 
    $<$<COMPILE_LANGUAGE:C>:-fPIC>
  )
elseif(CMAKE_C_COMPILER_ID MATCHES "Intel")
  target_compile_options(fms_unified PRIVATE 
    $<$<COMPILE_LANGUAGE:C>:-fPIC>
  )
endif()

message(STATUS "FMS configuration completed!")
message(STATUS "You can now run: cmake --build .")
message(STATUS "Note: Using most permissive syntax restriction settings")
