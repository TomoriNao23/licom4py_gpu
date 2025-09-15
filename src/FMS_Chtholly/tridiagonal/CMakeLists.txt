# Tridiagonal子目录的CMakeLists.txt

# 查找源文件
file(GLOB TRIDIAGONAL_SOURCES "*.F90")

# 创建三对角矩阵库
add_library(tridiagonal STATIC ${TRIDIAGONAL_SOURCES})

# 设置目标属性
set_target_properties(tridiagonal PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
)

# 设置包含目录
target_include_directories(tridiagonal PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

# 链接依赖库
target_link_libraries(tridiagonal PUBLIC constants)
