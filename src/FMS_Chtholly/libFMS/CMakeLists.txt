# libFMS子目录的CMakeLists.txt
# 这是最终打包FMS库的地方

# 创建最终的FMS库
add_library(FMS SHARED ${CMAKE_SOURCE_DIR}/include/file_version.h dummy.F90)

# 设置目标属性
set_target_properties(FMS PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  POSITION_INDEPENDENT_CODE ON
  VERSION ${PROJECT_VERSION}
  SOVERSION 2
)

# 设置包含目录
target_include_directories(FMS PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

# 链接所有子库
target_link_libraries(FMS PUBLIC
  platform
  constants
  tridiagonal
  mpp_parameter
  mpp
  mpp_data
  mpp_utilities
  mpp_domains
  mpp_io
  mpp_pset
  mpp_efp
  mpp_c
  mpp_memutils
  memutils
  fms_io
  fms
  # 暂时注释掉其他模块的库
  # affinity
  # mosaic
  # time_manager
  # diag_manager
  # drifters
  # axis_utils
  # horiz_interp
  # time_interp
  # column_diagnostics
  # block_control
  # data_override
  # astronomy
  # field_manager
  # coupler
  # diag_integral
  # monin_obukhov
  # interpolator
  # fft
  # amip_interp
  # oda_tools
  # exchange
  # topography
  # tracer_manager
  # station_data
  # sat_vapor_pres
  # random_numbers
)

# 在Apple系统上添加额外的库
if(APPLE)
  target_link_libraries(FMS PUBLIC memuse_c)
endif()

# 链接外部依赖
target_link_libraries(FMS PUBLIC
  MPI::MPI_C
  MPI::MPI_Fortran
  ${NETCDF_LIBRARIES}
  ${NETCDFF_LIBRARIES}
)

# 设置库的导出符号
set_target_properties(FMS PROPERTIES
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)
