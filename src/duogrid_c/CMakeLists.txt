cmake_minimum_required(VERSION 3.10)
project(FMS_Chtholly VERSION 1.0 LANGUAGES Fortran C)
# Chtholly <mengleshan@mail.iap.ac.cn> 
# Created: 2025.9.15
# Updated: 2025.9.20 (set FMS_Chtholly the minimum_required)

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
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O2 -fp-model source  -assume byterecl -ftz -traceback -mcmodel=medium")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -i4 -real-size 64 -no-prec-div -no-prec-sqrt")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fpp")
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
  # TODO other compiles
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
  
  # Set deployment target for MPI compilers to avoid version warnings
  if(APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmacosx-version-min=14.0")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -mmacosx-version-min=14.0")
  endif()
else()
  message(WARNING "MPI not found, proceeding without MPI support")
  set(MPI_FOUND FALSE)
endif()


# Set include directories


# Define preprocessor macros - use conditional definitions
if(MPI_FOUND)
  add_definitions(-Duse_libMPI)
endif()

# Detect operating system
if(APPLE)
  add_definitions(-D__APPLE__)
  # Set macOS deployment target to avoid version warnings
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum OS X deployment version")
  # Let CMake find the correct SDK automatically
  execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE MACOS_SDK_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(MACOS_SDK_PATH)
    set(CMAKE_OSX_SYSROOT ${MACOS_SDK_PATH})
    message(STATUS "Using macOS SDK: ${MACOS_SDK_PATH}")
  endif()
endif()

# Detect CRAY system
if(DEFINED ENV{CRAY_PRGENVINTEL} OR DEFINED ENV{CRAY_PRGENVGNU} OR DEFINED ENV{CRAY_PRGENVCRAY})
  add_definitions(-DCRAY)
endif()

# Fortran module output directory is already set above

# Create unified FMS library
add_library(duogrid_coordinate SHARED)

# Collect source files (removed fms and mpp)
file(GLOB_RECURSE FMS_SOURCES 
  "platform/*.F90"
  "chtholly/global_grid/*.F90"
  "chtholly/*.F90"
)

# Add chtholly source files after mpp (for dependency order)
file(GLOB CHTHOLLY_SOURCES "chtholly/*.F90")

# Merge all source files
set(FMS_ALL_SOURCES ${FMS_SOURCES} ${CHTHOLLY_SOURCES})

# Add source files to unified library
target_sources(duogrid_coordinate PRIVATE ${FMS_ALL_SOURCES})

# Set target properties
set_target_properties(duogrid_coordinate PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
)



# Link libraries - use conditional linking
if(MPI_FOUND)
  target_link_libraries(duogrid_coordinate PUBLIC MPI::MPI_C MPI::MPI_Fortran)
endif()



# Set Fortran compiler options
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  target_compile_options(duogrid_coordinate PRIVATE 
    $<$<COMPILE_LANGUAGE:Fortran>:-fdefault-real-8 -std=legacy -fallow-argument-mismatch -fcray-pointer -fno-implicit-none>
  )
#elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
#  target_compile_options(duogrid_coordinate PRIVATE 
#    $<$<COMPILE_LANGUAGE:Fortran>:-r8 -std-legacy -no-range-check>
#  )
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Cray")
  target_compile_options(duogrid_coordinate PRIVATE 
    $<$<COMPILE_LANGUAGE:Fortran>:-s real64 -h noomp -h noacc>
  )
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "PGI")
  target_compile_options(duogrid_coordinate PRIVATE 
    $<$<COMPILE_LANGUAGE:Fortran>:-r8 -i8 -Mfree -Mextend>
  )
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "NAG")
  target_compile_options(duogrid_coordinate PRIVATE 
    $<$<COMPILE_LANGUAGE:Fortran>:-double -free -f95>
  )
endif()

# Set C compiler options
if(CMAKE_C_COMPILER_ID MATCHES "GNU")
  target_compile_options(duogrid_coordinate PRIVATE 
    $<$<COMPILE_LANGUAGE:C>:-fPIC -w>
  )
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
  target_compile_options(duogrid_coordinate PRIVATE 
    $<$<COMPILE_LANGUAGE:C>:-fPIC -w -Wno-overriding-deployment-version>
  )
elseif(CMAKE_C_COMPILER_ID MATCHES "Intel")
  target_compile_options(duogrid_coordinate PRIVATE 
    $<$<COMPILE_LANGUAGE:C>:-fPIC -w>
  )
endif()

message(STATUS "FMS configuration completed!")
message(STATUS "You can now run: cmake --build .")
message(STATUS "Note: Using most permissive syntax restriction settings")