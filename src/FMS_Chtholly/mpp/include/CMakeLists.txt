# MPP Include子目录的CMakeLists.txt

# 安装头文件
install(FILES
  mpp.h
  mpp_domains.h
  mpp_io.h
  mpp_parameter.h
  mpp_pset.h
  mpp_util.h
  DESTINATION include/mpp
)

# 设置包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
