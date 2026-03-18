module Chtholly_c_wrapper

  use fms_mod,            only: fms_init, fms_end

  use mpp_mod,            only: mpp_npes, mpp_pe
  use mpp_mod,            only: mpp_get_current_pelist
  
  use mpp_domains_mod,    only: domain2d
  use mpp_domains_mod,    only: mpp_define_mosaic
  use mpp_domains_mod,    only: mpp_get_compute_domain 
  use mpp_domains_mod,    only: mpp_get_data_domain
  use mpp_domains_mod,    only: mpp_get_boundary
  use mpp_domains_mod,    only: mpp_update_domains
  use mpp_domains_mod,    only: CGRID_NE

implicit none

public :: fmsinit, fmsend, ext_scalar_2d, communication2d

type(domain2d),public :: domain
integer, allocatable, dimension(:),public :: pelist

integer, public :: isd, ied, jsd, jed
integer, public :: is, ie, js, je
integer, public :: xsize, ysize
integer, public :: npx, npy
integer :: tile


contains
  subroutine fmsinit(nx,ny,layout1,layout2) bind(C, name="chtholly_init")
    use iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: nx
    integer(c_int), value :: ny
    integer(c_int), value :: layout1
    integer(c_int), value :: layout2
    call fms_init()
    allocate ( pelist(mpp_npes()) )
    call mpp_get_current_pelist(pelist)
    call define_cube(                             &
                     int(nx,kind=kind(isd)),      &
                     int(ny,kind=kind(isd)),      &
                     int(layout1,kind=kind(isd)), &
                     int(layout2,kind=kind(isd))  &
                    )
    npx = nx + 1
    npy = ny + 1
  end subroutine

  subroutine fmsend() bind(C, name="chtholly_end")
    implicit none
    deallocate ( pelist )
    call fms_end()
  end subroutine

  subroutine define_cube(nx,ny,layout1,layout2)
    implicit none
    integer,intent(in) :: nx, ny, layout1, layout2
    integer, parameter :: num_contact = 12, nregions = 6
    integer, dimension(nregions)    :: pe_start, pe_end
    integer, dimension(num_contact) :: &
        tile1, tile2, &
        istart1, iend1, jstart1, jend1, &
        istart2, iend2, jstart2, jend2
    integer, dimension(2,nregions)  :: layout2D
    integer, dimension(4,nregions)  :: global_indices
    integer :: npes_per_tile
    integer :: ng, n
    integer :: layout(2)

    ng  = 3
    layout=[layout1,layout2]
    npes_per_tile = layout(1)*layout(2)

    do n = 1, nregions
        global_indices(:,n) = [1,nx,1,ny]
        layout2D(:,n)         = layout
        pe_start(n) = pelist(1) + (n-1)*npes_per_tile
        pe_end(n)   = pe_start(n) + npes_per_tile -1
    end do

  tile1(1) = 1; tile2(1) = 2
  istart1(1) = nx; iend1(1) = nx; jstart1(1) = 1;  jend1(1) = ny
  istart2(1) = 1;  iend2(1) = 1;  jstart2(1) = 1;  jend2(1) = ny
  !--- Contact line 2, between tile 1 (NORTH) and tile 3 (WEST)
  tile1(2) = 1; tile2(2) = 3
  istart1(2) = 1;  iend1(2) = nx; jstart1(2) = ny; jend1(2) = ny
  istart2(2) = 1;  iend2(2) = 1;  jstart2(2) = ny; jend2(2) = 1
  !--- Contact line 3, between tile 1 (WEST) and tile 5 (NORTH)
  tile1(3) = 1; tile2(3) = 5
  istart1(3) = 1;  iend1(3) = 1;  jstart1(3) = 1;  jend1(3) = ny
  istart2(3) = nx; iend2(3) = 1;  jstart2(3) = ny; jend2(3) = ny
  !--- Contact line 4, between tile 1 (SOUTH) and tile 6 (NORTH)
  tile1(4) = 1; tile2(4) = 6
  istart1(4) = 1;  iend1(4) = nx; jstart1(4) = 1;  jend1(4) = 1
  istart2(4) = 1;  iend2(4) = nx; jstart2(4) = ny; jend2(4) = ny
  !--- Contact line 5, between tile 2 (NORTH) and tile 3 (SOUTH)
  tile1(5) = 2; tile2(5) = 3
  istart1(5) = 1;  iend1(5) = nx; jstart1(5) = ny; jend1(5) = ny
  istart2(5) = 1;  iend2(5) = nx; jstart2(5) = 1;  jend2(5) = 1
  !--- Contact line 6, between tile 2 (EAST) and tile 4 (SOUTH)
  tile1(6) = 2; tile2(6) = 4
  istart1(6) = nx; iend1(6) = nx; jstart1(6) = 1;  jend1(6) = ny
  istart2(6) = nx; iend2(6) = 1;  jstart2(6) = 1;  jend2(6) = 1
  !--- Contact line 7, between tile 2 (SOUTH) and tile 6 (EAST)
  tile1(7) = 2; tile2(7) = 6
  istart1(7) = 1;  iend1(7) = nx; jstart1(7) = 1;  jend1(7) = 1
  istart2(7) = nx; iend2(7) = nx; jstart2(7) = ny; jend2(7) = 1
  !--- Contact line 8, between tile 3 (EAST) and tile 4 (WEST)
  tile1(8) = 3; tile2(8) = 4
  istart1(8) = nx; iend1(8) = nx; jstart1(8) = 1;  jend1(8) = ny
  istart2(8) = 1;  iend2(8) = 1;  jstart2(8) = 1;  jend2(8) = ny
  !--- Contact line 9, between tile 3 (NORTH) and tile 5 (WEST)
  tile1(9) = 3; tile2(9) = 5
  istart1(9) = 1;  iend1(9) = nx; jstart1(9) = ny; jend1(9) = ny
  istart2(9) = 1;  iend2(9) = 1;  jstart2(9) = ny; jend2(9) = 1
  !--- Contact line 10, between tile 4 (NORTH) and tile 5 (SOUTH)
  tile1(10) = 4; tile2(10) = 5
  istart1(10) = 1;  iend1(10) = nx; jstart1(10) = ny; jend1(10) = ny
  istart2(10) = 1;  iend2(10) = nx; jstart2(10) = 1;  jend2(10) = 1
  !--- Contact line 11, between tile 4 (EAST) and tile 6 (SOUTH)
  tile1(11) = 4; tile2(11) = 6
  istart1(11) = nx; iend1(11) = nx; jstart1(11) = 1;  jend1(11) = ny
  istart2(11) = nx; iend2(11) = 1;  jstart2(11) = 1;  jend2(11) = 1
  !--- Contact line 12, between tile 5 (EAST) and tile 6 (WEST)
  tile1(12) = 5; tile2(12) = 6
  istart1(12) = nx; iend1(12) = nx; jstart1(12) = 1;  jend1(12) = ny
  istart2(12) = 1;  iend2(12) = 1;  jstart2(12) = 1;  jend2(12) = ny

  !--- mpp define mosaic
  call mpp_define_mosaic( &
      global_indices, layout2D, domain, &
      nregions, num_contact, &
      tile1, tile2, &
      istart1, iend1, jstart1, jend1, &
      istart2, iend2, jstart2, jend2, &
      pe_start=pe_start, pe_end=pe_end, symmetry=.true., &
      shalo = 3, nhalo = 3, whalo = 3, ehalo = 3, &
      name = "cube")

    tile = (mpp_pe()-pelist(1))/npes_per_tile+1 

    !--- set dimensions
    call mpp_get_compute_domain( domain, is,  ie,  js,  je  )
    call mpp_get_data_domain   ( domain, isd, ied, jsd, jed )
    xsize = ied - isd + 1
    ysize = jed - jsd + 1
  end subroutine

  subroutine ext_scalar_2d(field) bind(C, name="chtholly_ext_scalar_2d")
    use iso_c_binding, only: c_double
    implicit none
    real(c_double), intent(inout) :: field(*)
    real(c_double), allocatable :: field_2d(:,:)

    ! Reshape the 1D array to 2D with Fortran ordering
    allocate(field_2d(xsize, ysize))
    field_2d = transpose(reshape(field(1:xsize*ysize), [ysize, xsize]))

    call mpp_update_domains(field_2d, domain)

    ! Copy back to 1D array
    field(1:xsize*ysize) = reshape(transpose(field_2d), [xsize*ysize])
    deallocate(field_2d)

  end subroutine ext_scalar_2d

  subroutine communication2d(field1,field2) bind(C, name="chtholly_communication2d")
    !> @brief Boundary Communication of the Flux

    !import
      use iso_c_binding, only: c_double
      implicit none
      real(c_double), intent(inout) :: field1(*), field2(*)
      real(c_double), dimension(:,:), allocatable :: flux_xx, flux_yy
      real(c_double), dimension(:,:), allocatable :: flux_x, flux_y
      real(c_double), dimension(:), allocatable :: wbuffer, ebuffer, sbuffer, nbuffer
    !alloca()
      allocate(flux_xx(isd:ied,jsd:jed))
      allocate(flux_yy(isd:ied,jsd:jed))
      allocate(flux_x(isd:ied+1,jsd:jed))
      allocate(flux_y(isd:ied,jsd:jed+1))
      allocate (sbuffer(npx+2),nbuffer(npx+2),wbuffer(npy+2),ebuffer(npy+2))
      flux_x = 0
      flux_y = 0
    !cal
      flux_xx = transpose(reshape(field1(1:xsize*ysize), [ysize, xsize]))
      flux_yy = transpose(reshape(field2(1:xsize*ysize), [ysize, xsize]))
      flux_x(isd:ied,jsd:jed) = flux_xx(isd:ied,jsd:jed)
      flux_y(isd:ied,jsd:jed) = flux_yy(isd:ied,jsd:jed)
    !update the boundary
      call mpp_get_boundary(flux_x, flux_y, domain, wbufferx=wbuffer, &
          ebufferx=ebuffer, sbuffery=sbuffer, nbuffery=nbuffer, gridtype=CGRID_NE)
      flux_y(is:ie,js)   = 0.5*(flux_y(is:ie,js)   + sbuffer(1:ie-is+1))
      flux_y(is:ie,je+1) = 0.5*(flux_y(is:ie,je+1) + nbuffer(1:ie-is+1))    
      flux_x(is,js:je)   = 0.5*(flux_x(is,js:je)   + wbuffer(1:je-js+1))
      flux_x(ie+1,js:je) = 0.5*(flux_x(ie+1,js:je) + ebuffer(1:je-js+1))
    !reshape
      flux_xx(isd:ied,jsd:jed) = flux_x(isd:ied,jsd:jed)
      flux_yy(isd:ied,jsd:jed) = flux_y(isd:ied,jsd:jed)
      field1(1:xsize*ysize) = reshape(transpose(flux_xx), [xsize*ysize])
      field2(1:xsize*ysize) = reshape(transpose(flux_yy), [xsize*ysize])
    !dealloca()
      deallocate(flux_x,flux_y,flux_xx,flux_yy,sbuffer,nbuffer,wbuffer,ebuffer)
      
  end subroutine

end module Chtholly_c_wrapper