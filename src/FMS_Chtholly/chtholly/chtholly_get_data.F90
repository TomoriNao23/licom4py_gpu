module Chtholly_get_data

  use Chtholly_c_wrapper, only: isd, ied, jsd, jed
  use Chtholly_c_wrapper, only: is, ie, js, je
  use Chtholly_c_wrapper, only: xsize, ysize
  use Chtholly_c_wrapper, only: tile

  use Chtholly_c_wrapper, only: chtholly_init => fmsinit
  use Chtholly_c_wrapper, only: chtholly_end => fmsend
  use Chtholly_c_wrapper, only: chtholly_ext_scalar_2d => ext_scalar_2d

  use iso_c_binding, only: c_int, c_double

  implicit none

  private

  public :: chtholly_get_tile
  public :: chtholly_get_xsize
  public :: chtholly_get_ysize
  public :: chtholly_get_mpp_pe
  public :: chtholly_get_is
  public :: chtholly_get_ie
  public :: chtholly_get_js
  public :: chtholly_get_je
  public :: chtholly_get_isd
  public :: chtholly_get_ied
  public :: chtholly_get_jsd
  public :: chtholly_get_jed

  contains

  ! Chtholly's functions

    function chtholly_get_tile() bind(C, name="chtholly_get_tile")
      implicit none
      integer(c_int) :: chtholly_get_tile
      chtholly_get_tile = tile
    end function chtholly_get_tile

    function chtholly_get_xsize() bind(C, name="chtholly_get_xsize")
      implicit none
      integer(c_int) :: chtholly_get_xsize
      chtholly_get_xsize = xsize
    end function chtholly_get_xsize

    function chtholly_get_ysize() bind(C, name="chtholly_get_ysize")
      implicit none
      integer(c_int) :: chtholly_get_ysize
      chtholly_get_ysize = ysize
    end function chtholly_get_ysize

    function chtholly_get_is() bind(C, name="chtholly_get_is")
      implicit none
      integer(c_int) :: chtholly_get_is
      chtholly_get_is = is
    end function chtholly_get_is

    function chtholly_get_ie() bind(C, name="chtholly_get_ie")
      implicit none
      integer(c_int) :: chtholly_get_ie
      chtholly_get_ie = ie
    end function chtholly_get_ie

    function chtholly_get_js() bind(C, name="chtholly_get_js")
      implicit none
      integer(c_int) :: chtholly_get_js
      chtholly_get_js = js
    end function chtholly_get_js

    function chtholly_get_je() bind(C, name="chtholly_get_je")
      implicit none
      integer(c_int) :: chtholly_get_je
      chtholly_get_je = je
    end function chtholly_get_je

    function chtholly_get_isd() bind(C, name="chtholly_get_isd")
      implicit none
      integer(c_int) :: chtholly_get_isd
      chtholly_get_isd = isd
    end function chtholly_get_isd

    function chtholly_get_ied() bind(C, name="chtholly_get_ied")
      implicit none
      integer(c_int) :: chtholly_get_ied
      chtholly_get_ied = ied
    end function chtholly_get_ied

    function chtholly_get_jsd() bind(C, name="chtholly_get_jsd")
      implicit none
      integer(c_int) :: chtholly_get_jsd
      chtholly_get_jsd = jsd
    end function chtholly_get_jsd

    function chtholly_get_jed() bind(C, name="chtholly_get_jed")
      implicit none
      integer(c_int) :: chtholly_get_jed
      chtholly_get_jed = jed
    end function chtholly_get_jed

  ! MPP's functions

    function chtholly_get_mpp_pe() bind(C, name="chtholly_get_mpp_pe")
      use mpp_mod, only: mpp_pe
      implicit none
      integer(c_int) :: chtholly_get_mpp_pe
      chtholly_get_mpp_pe = mpp_pe()
    end function chtholly_get_mpp_pe


end module Chtholly_get_data