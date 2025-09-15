# Platform子目录的CMakeLists.txt

# 查找源文件
file(GLOB PLATFORM_SOURCES "*.F90")

# 创建平台库
add_library(platform STATIC ${PLATFORM_SOURCES})

# 设置目标属性
set_target_properties(platform PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
)

# 设置包含目录
target_include_directories(platform PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

# 链接MPI
target_link_libraries(platform PUBLIC MPI::MPI_C)

# 设置Fortran编译器选项
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  target_compile_options(platform PRIVATE -fdefault-real-8)
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  target_compile_options(platform PRIVATE -r8)
endif()
