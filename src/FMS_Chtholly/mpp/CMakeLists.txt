# MPP子目录的CMakeLists.txt

# 添加include子目录
add_subdirectory(include)

# 查找源文件
file(GLOB MPP_SOURCES "*.F90" "*.c")

# 创建MPP库
add_library(mpp STATIC ${MPP_SOURCES})

# 设置目标属性
set_target_properties(mpp PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
)

# 设置包含目录
target_include_directories(mpp PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/mpp/include
)

# 链接依赖库
target_link_libraries(mpp PUBLIC 
  platform
  constants
  MPI::MPI_C
  MPI::MPI_Fortran
)

# 设置Fortran编译器选项
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  target_compile_options(mpp PRIVATE -fdefault-real-8)
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  target_compile_options(mpp PRIVATE -r8)
endif()

# 创建MPP参数库
add_library(mpp_parameter STATIC mpp_parameter.F90)
set_target_properties(mpp_parameter PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(mpp_parameter PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mpp_parameter PUBLIC platform)

# 创建MPP数据库
add_library(mpp_data STATIC mpp_data.F90)
set_target_properties(mpp_data PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(mpp_data PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mpp_data PUBLIC mpp_parameter)

# 创建MPP工具库
add_library(mpp_utilities STATIC mpp_utilities.F90)
set_target_properties(mpp_utilities PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(mpp_utilities PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mpp_utilities PUBLIC mpp mpp_efp)

# 创建MPP内存工具库
add_library(mpp_memutils STATIC mpp_memutils.F90)
set_target_properties(mpp_memutils PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(mpp_memutils PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mpp_memutils PUBLIC mpp)

# 创建MPP进程集库
add_library(mpp_pset STATIC mpp_pset.F90)
set_target_properties(mpp_pset PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(mpp_pset PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mpp_pset PUBLIC mpp)

# 创建MPP EFP库
add_library(mpp_efp STATIC mpp_efp.F90)
set_target_properties(mpp_efp PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(mpp_efp PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mpp_efp PUBLIC mpp_parameter mpp)

# 创建MPP域库
add_library(mpp_domains STATIC mpp_domains.F90)
set_target_properties(mpp_domains PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(mpp_domains PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mpp_domains PUBLIC mpp_data mpp_parameter mpp mpp_memutils mpp_pset mpp_efp)

# 创建MPP IO库
add_library(mpp_io STATIC mpp_io.F90)
set_target_properties(mpp_io PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(mpp_io PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mpp_io PUBLIC mpp_parameter mpp mpp_domains)

# 创建MPP C库
add_library(mpp_c STATIC nsclock.c threadloc.c)
set_target_properties(mpp_c PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(mpp_c PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mpp_c PUBLIC MPI::MPI_C)
