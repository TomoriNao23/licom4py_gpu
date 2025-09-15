!-------------------------------------------------------------------------------
!> @brief create duogrid remap region local coordinates
!> @author Xi.Chen <xic@princeton.edu>
!> @date 01/15/2020
!
!  REVISION HISTORY:
!  01/15/2020 - Initial Version
!-------------------------------------------------------------------------------

module global_grid_gen_coords_mod

    !------ fms modules
    use constants_mod,          only: pi=>pi_8
    !------ AC modules
    use global_grid_data_mod,   only: global_grid_type
    use lib_grid_mod,           only: R_GRID
    use lib_grid_mod,           only: lib_great_circ_dist

    implicit none
    private

    public :: global_grid_gen_coords

contains
    !===========================================================================
    !> @brief create duogrid remap region local coordinates
    subroutine global_grid_gen_coords(gg)
        type(global_grid_type), intent(inout) :: gg
        !--- local
        integer :: i, j, k, nc, n, ng
        integer :: isd, ied, jsd, jed, is, ie, js, je
        real(kind=R_GRID), dimension(:),   allocatable :: line
        real(kind=R_GRID), dimension(:,:), allocatable :: pt

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
        allocate( line(isd:ied) )
        allocate( pt(2,isd:ied) )

        line(:) = -999.

        do k = 1, ng

            !--- assign one layer of rmp kik pts
            pt(:,:) = gg%pt_kik(:,is-k,:,1)

            !--- get coords
            line(nc) = 0.
            do j = js, nc-1
!                line(j) = - lib_great_circ_dist( &
!                    pt(:,j), pt(:,nc))
                line(j) = pt(2,j) - pt(2,nc) !--- take advantage of meridians
                line(je-j+1) = -line(j)
            enddo

            !--- populate the coords to cubed sphere
            do n = 1, 6
                do j = js,je
                    !--- west and south
                    i = is-k
                    gg%kik_y(i,j,n) = line(j)
                    gg%kik_x(j,i,n) = line(j)
                    !--- east and north
                    i = ie+k
                    gg%kik_y(i,j,n) = line(j)
                    gg%kik_x(j,i,n) = line(j)
                enddo
            enddo

            !--- assign one layer of rmp ext pts
            pt(:,:) = gg%pt_ext(:,is-k,:,1)

            !--- get coords
            line(nc) = 0.
            do j = jsd, nc-1
!                line(j) = - lib_great_circ_dist( &
!                    pt(:,j), pt(:,nc))
                line(j) = pt(2,j) - pt(2,nc) !--- take advantage of meridians
                line(je-j+1) = -line(j)
            enddo

            !--- populate the coords to cubed sphere
            do n = 1, 6
                do j = jsd,jed
                    !--- west and south
                    i = is-k
                    gg%ext_y(i,j,n) = line(j)
                    gg%ext_x(j,i,n) = line(j)
                    !--- east and north
                    i = ie+k
                    gg%ext_y(i,j,n) = line(j)
                    gg%ext_x(j,i,n) = line(j)
                enddo
            enddo

        enddo

        !--- deallocate arrays
        deallocate( line )
        deallocate( pt )


    end subroutine global_grid_gen_coords
    !===========================================================================
end module global_grid_gen_coords_mod

