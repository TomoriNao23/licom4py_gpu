!-------------------------------------------------------------------------------
!> @brief generate dx, dy, area, etc.
!> @author Xi.Chen <xic@princeton.edu>
!> @date 01/17/2020
!
!  REVISION HISTORY:
!  01/17/2020 - Initial Version
!-------------------------------------------------------------------------------

module global_grid_gen_cell_mod

    !------ fms modules
    use constants_mod,          only: pi=>pi_8
    !------ AC modules
    use global_grid_data_mod,   only: global_grid_type
    use lib_grid_mod,           only: R_GRID, RADIUS
    use lib_grid_mod,           only: lib_4pt_area

    implicit none
    private

    public :: global_grid_gen_ds
    public :: global_grid_gen_da

contains
    !===========================================================================
    !> @brief generate dx, dy via lat, leveraging symmetry.
    !         (should use great circle in stretched case)
    subroutine global_grid_gen_ds(gg)
        type(global_grid_type), intent(inout) :: gg
        !--- local
        integer :: i, j, k, nc, n, ng
        integer :: ii, jj
        integer :: isd, ied, jsd, jed, is, ie, js, je
        real(R_GRID), dimension(:,:,:), allocatable :: pt
        real(R_GRID), dimension(:,:), allocatable :: dy

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
        allocate( pt(2, isd:ied, jsd:jed) )
        allocate( dy(isd:nc, jsd:nc-1) ) !--- intentional dims to reveal errors

        !--- assign tile 1 ext
        pt(:,:,:) = gg%pt_ext(:,:,:,1)

        !--- gen ds ext
        do j = jsd,nc-1
            do i = isd,nc
                dy(i,j) = (pt(2,i,j+1)-pt(2,i,j)) * RADIUS
            enddo
        enddo

        !--- populate to gg (both ext and kik, replace kik halo ds later)
        do n = 1, 6
            do j = jsd, nc-1
                do i = isd, nc
                    !--- dy
                    ii = ie-i+1
                    jj = je-j
                    gg%ext_dy(i, j, n) = dy(i,j) ! sw
                    gg%ext_dy(ii,j, n) = dy(i,j) ! se
                    gg%ext_dy(i, jj,n) = dy(i,j) ! nw
                    gg%ext_dy(ii,jj,n) = dy(i,j) ! ne
                    gg%kik_dy(i, j, n) = dy(i,j) ! sw
                    gg%kik_dy(ii,j, n) = dy(i,j) ! se
                    gg%kik_dy(i, jj,n) = dy(i,j) ! nw
                    gg%kik_dy(ii,jj,n) = dy(i,j) ! ne
                    !--- dx
                    gg%ext_dx(j, i, n) = dy(i,j) ! sw
                    gg%ext_dx(j, ii,n) = dy(i,j) ! nw
                    gg%ext_dx(jj,i, n) = dy(i,j) ! se
                    gg%ext_dx(jj,ii,n) = dy(i,j) ! ne
                    gg%kik_dx(j, i, n) = dy(i,j) ! sw
                    gg%kik_dx(j, ii,n) = dy(i,j) ! nw
                    gg%kik_dx(jj,i, n) = dy(i,j) ! se
                    gg%kik_dx(jj,ii,n) = dy(i,j) ! ne
                enddo
            enddo
        enddo

        !--- assign tile1 kik for halo
        pt(:,:,:) = gg%pt_kik(:,:,:,1)

        !--- gen ds kik
        do j = js,nc-1
            do i = isd,is-1
                dy(i,j) = (pt(2,i,j+1)-pt(2,i,j)) * RADIUS
            enddo
        enddo

        !--- populate to gg kik halo ds
        do n = 1, 6
            do j = js, nc-1
                do i = isd, is-1
                    !--- dy
                    ii = ie-i+1
                    jj = je-j
                    gg%kik_dy(i, j, n) = dy(i,j) ! sw
                    gg%kik_dy(ii,j, n) = dy(i,j) ! se
                    gg%kik_dy(i, jj,n) = dy(i,j) ! nw
                    gg%kik_dy(ii,jj,n) = dy(i,j) ! ne
                    !--- dx
                    gg%kik_dx(j, i, n) = dy(i,j) ! sw
                    gg%kik_dx(j, ii,n) = dy(i,j) ! nw
                    gg%kik_dx(jj,i, n) = dy(i,j) ! se
                    gg%kik_dx(jj,ii,n) = dy(i,j) ! ne
                enddo
            enddo
        enddo

        !--- deallocate arrays
        deallocate( pt )
        deallocate( dy )

    end subroutine global_grid_gen_ds
    !===========================================================================
    !> @brief generate da
    subroutine global_grid_gen_da(gg)
        type(global_grid_type), intent(inout) :: gg
        !--- local
        integer :: i, j, k, nc, n, ng
        integer :: ii, jj
        integer :: isd, ied, jsd, jed, is, ie, js, je
        real(R_GRID), dimension(:,:,:), allocatable :: pt
        real(R_GRID), dimension(:,:), allocatable :: da

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
        allocate( pt(2, isd:ied, jsd:jed) )
        allocate( da(isd:nc-1, jsd:nc-1) ) !--- intentional dims to reveal errors

        !--- assign tile 1 ext
        pt(:,:,:) = gg%pt_ext(:,:,:,1)

        !--- gen da ext
        do j = jsd,nc-1
            do i = isd,nc-1
                da(i,j) = lib_4pt_area( &
                    pt(:,i,j), pt(:,i+1,j), pt(:,i+1,j+1), pt(:,i,j+1), &
                    RADIUS)
            enddo
        enddo

        !--- populate to gg (both ext and kik, replace kik halo ds later)
        do n = 1, 6
            do j = jsd, nc-1
                do i = isd, nc-1
                    ii = ie-i
                    jj = je-j
                    gg%ext_da(i, j, n) = da(i,j) ! sw
                    gg%ext_da(ii,j, n) = da(i,j) ! se
                    gg%ext_da(i, jj,n) = da(i,j) ! nw
                    gg%ext_da(ii,jj,n) = da(i,j) ! ne
                    gg%kik_da(i, j, n) = da(i,j) ! sw
                    gg%kik_da(ii,j, n) = da(i,j) ! se
                    gg%kik_da(i, jj,n) = da(i,j) ! nw
                    gg%kik_da(ii,jj,n) = da(i,j) ! ne
                enddo
            enddo
        enddo

        !--- assign tile1 kik for halo
        pt(:,:,:) = gg%pt_kik(:,:,:,1)

        !--- gen ds ext
        do j = js,nc-1
            do i = isd,is-1
                da(i,j) = lib_4pt_area( &
                    pt(:,i,j), pt(:,i+1,j), pt(:,i+1,j+1), pt(:,i,j+1), &
                    RADIUS)
            enddo
        enddo

        !--- populate to gg (both ext and kik, replace kik halo ds later)
        do n = 1, 6
            do j = js, nc-1
                do i = isd, is-1
                    !--- y-dir
                    ii = ie-i
                    jj = je-j
                    gg%kik_da(i, j, n) = da(i,j) ! sw
                    gg%kik_da(ii,j, n) = da(i,j) ! se
                    gg%kik_da(i, jj,n) = da(i,j) ! nw
                    gg%kik_da(ii,jj,n) = da(i,j) ! ne
                    !--- x-dir
                    gg%kik_da(j, i, n) = da(i,j) ! sw
                    gg%kik_da(j, ii,n) = da(i,j) ! nw
                    gg%kik_da(jj,i, n) = da(i,j) ! se
                    gg%kik_da(jj,ii,n) = da(i,j) ! ne
                enddo
            enddo
        enddo

        !--- deallocate arrays
        deallocate( pt )
        deallocate( da )

    end subroutine global_grid_gen_da
    !=============================================================================
end module global_grid_gen_cell_mod

