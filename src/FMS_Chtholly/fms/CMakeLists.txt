# FMS子目录的CMakeLists.txt

# 查找源文件
file(GLOB FMS_SOURCES "*.F90")

# 创建FMS IO库
add_library(fms_io STATIC ${FMS_SOURCES})

# 设置目标属性
set_target_properties(fms_io PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
)

# 设置包含目录
target_include_directories(fms_io PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/mpp/include
)

# 链接依赖库
target_link_libraries(fms_io PUBLIC 
  mpp
  mpp_io
  ${NETCDF_LIBRARIES}
  ${NETCDFF_LIBRARIES}
)

# 创建主FMS库
add_library(fms STATIC fms.F90)

# 设置目标属性
set_target_properties(fms PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
)

# 设置包含目录
target_include_directories(fms PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

# 链接依赖库
target_link_libraries(fms PUBLIC fms_io)
