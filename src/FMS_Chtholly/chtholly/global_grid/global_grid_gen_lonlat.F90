!-------------------------------------------------------------------------------
!> @brief genarate global grid
!> @author Xi.Chen <xic@princeton.edu>
!> @date 01/14/2020
!
!  REVISION HISTORY:
!  01/14/2020 - Initial Version
!-------------------------------------------------------------------------------

module global_grid_gen_lonlat_mod

    !------ fms modules
    !------ AC modules
    use global_grid_data_mod,   only: global_grid_type
    use global_grid_util_mod,   only: global_grid_update_kik
    use lib_grid_mod,           only: pi
    use lib_grid_mod,           only: R_GRID
    use lib_grid_mod,           only: lib_cart2lonlat

    implicit none
    private

    public :: global_grid_gen_lonlat_equal_edge

contains
    !===========================================================================
    !> @brief get lonlat values of the global grid
    subroutine global_grid_gen_lonlat_equal_edge(gg)
        type(global_grid_type), intent(inout) :: gg
        !--- local
        integer :: i, j, ii, jj, nc, n, ng
        integer :: isd, ied, jsd, jed, is, ie, js, je
        real(kind=R_GRID) :: rsq3, rsq2, alpha, dela
        real(kind=R_GRID) :: x, y, angl_x, angl_y, angl_g, angl, sina, cosa
        real(kind=R_GRID), dimension(:,:,:,:), allocatable :: cart
        real(kind=R_GRID), dimension(:),   allocatable :: line
        real(kind=R_GRID), parameter :: shift_fac = -18.
        real(kind=R_GRID), dimension(3,3) :: rot_z

        !--- assign dims
        is = 1
        ie = 2*gg%res+1
        js = is
        je = ie

        ng = 2*gg%ng

        isd = is - ng
        ied = ie + ng
        jsd = isd
        jed = ied

        nc = 1+gg%res

        !--- allocate arrays
        allocate( cart(3, isd:ied, jsd:jed, 6) )
        allocate( line(isd:ied) )

        !------ xic: equal_angular part
        if (gg%grid_type==2) then
            !--- get the angular step
            alpha = pi/4.0
            dela = 2.d0*alpha / real(ie-is,kind=R_GRID)

            !--- form half line: computational domain
            line(:) = -999.
            line(js) = -1.
            line(nc) = 0.
            do j = js+1, nc-1
                angl_y = (j-1)*dela - alpha
                y = tan(angl_y)
                line(j) = y
            enddo
        endif
        !------ xic: end equal_edge part

        !------ xic: equal_edge part
        if (gg%grid_type==0) then
            !--- get the angular step
            rsq3 = 1.d0/sqrt(3.d0)
            rsq2 = 1.d0/sqrt(2.d0)
            alpha = asin( rsq3 )
            dela = 2.d0*alpha / real(ie-is,kind=R_GRID)

            !--- form half line: computational domain
            line(:) = -999.
            line(js) = -1.
            line(nc) = 0.
            do j = js+1, nc-1
                angl_y = (j-1)*dela - alpha
                y = tan(angl_y)*sqrt(2.d0)
                line(j) = y
            enddo
        endif
        !------ xic: end equal_edge part

        !--- form half line: ghost cells
        do j = jsd, js-1
            jj = 2*js - j
            angl_g = -0.5*pi - atan(line(jj))
            line(j) = tan(angl_g)
        enddo

        !--- form full line
        do j = jsd, nc-1
            jj = je-j+1
            line(jj) = -line(j)
        enddo

        !--- populate to 6 tiles cart
        do j = jsd, jed
            do i = isd, ied
                x = line(i)
                y = line(j)
                cart(:, i, j, 1) = [1.0, x, y]
                cart(:, i, j, 2) = [-x, 1.0, y]
                cart(:, i, j, 3) = [-x, -y, 1.0]
                cart(:, i, j, 4) = [-1.0, -y, -x]
                cart(:, i, j, 5) = [y, -1.0, -x]
                cart(:, i, j, 6) = [y, x, -1.0]
            enddo
        enddo

        !--- Shift the corner away from Japan
!        angl = pi/shift_fac
!        sina = sin(angl)
!        cosa = cos(angl)
!        rot_z(:,:) = 0.
!        rot_z(1,1) = cosa
!        rot_z(2,2) = cosa
!        rot_z(3,3) = 1.
!        rot_z(1,2) = -sina
!        rot_z(2,1) = sina
!
!        do n = 1,6
!            do j = jsd, jed
!                do i = isd, ied
!
!                    cart(:,i,j,n) = matmul(rot_z, cart(:,i,j,n))
!
!                enddo
!            enddo
!        enddo

        !--- populate to 6 tiles lonlat
        do n = 1,6
            do j = jsd, jed
                do i = isd, ied
                    gg%pt_ext(:, i, j, n) = lib_cart2lonlat(cart(:,i,j,n))
                enddo
            enddo
        enddo

        call global_grid_update_kik(gg%pt_ext, gg%pt_kik, isd, ied, ng, 2)

        !--- deallocate arrays
        deallocate( cart )
        deallocate( line )

    end subroutine global_grid_gen_lonlat_equal_edge
    !===========================================================================
end module global_grid_gen_lonlat_mod


