# 通用子目录CMakeLists.txt模板
# 使用方法：复制此文件到子目录，并修改相应的变量

# 设置库名称（需要修改）
set(LIB_NAME "your_library_name")

# 查找源文件
file(GLOB LIB_SOURCES "*.F90" "*.c")

# 创建库
add_library(${LIB_NAME} STATIC ${LIB_SOURCES})

# 设置目标属性
set_target_properties(${LIB_NAME} PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
)

# 设置包含目录
target_include_directories(${LIB_NAME} PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/mpp/include
)

# 链接依赖库（需要根据实际情况修改）
target_link_libraries(${LIB_NAME} PUBLIC
  mpp
  # 添加其他依赖库
)

# 设置Fortran编译器选项
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  target_compile_options(${LIB_NAME} PRIVATE -fdefault-real-8)
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  target_compile_options(${LIB_NAME} PRIVATE -r8)
endif()
