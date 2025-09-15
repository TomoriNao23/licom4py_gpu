# Memutils子目录的CMakeLists.txt

# 查找源文件
file(GLOB MEMUTILS_SOURCES "*.F90" "*.c")

# 创建内存工具库
add_library(memutils STATIC ${MEMUTILS_SOURCES})

# 设置目标属性
set_target_properties(memutils PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
)

# 设置包含目录
target_include_directories(memutils PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

# 链接依赖库
target_link_libraries(memutils PUBLIC mpp)

# 创建C内存使用库（仅在Apple系统上）
if(APPLE)
  add_library(memuse_c STATIC memuse.c)
  set_target_properties(memuse_c PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(memuse_c PUBLIC
    ${CMAKE_SOURCE_DIR}/include
  )
endif()
