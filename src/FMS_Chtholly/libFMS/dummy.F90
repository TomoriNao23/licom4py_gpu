! Dummy source file for FMS library
! This file is needed to ensure CMake can determine the linker language

module fms_dummy
  implicit none
  private
  public :: fms_dummy_init
  
contains
  
  subroutine fms_dummy_init()
    ! This is a dummy initialization routine
    ! It does nothing but ensures the module is not empty
    return
  end subroutine fms_dummy_init
  
end module fms_dummy
