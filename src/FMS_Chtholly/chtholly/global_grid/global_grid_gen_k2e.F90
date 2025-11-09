!-------------------------------------------------------------------------------
!> @brief init kinked to extended grid remap coefficients
!> @author Xi.Chen <xic@princeton.edu>
!> @date 01/24/2020
!
!  REVISION HISTORY:
!  01/24/2020 - Initial Version
!-------------------------------------------------------------------------------

module global_grid_gen_k2e_mod

    !------ fms modules
    !------ AC modules
    use global_grid_data_mod,   only: global_grid_type
    use lib_grid_mod,           only: R_GRID
    use lib_grid_mod,           only: lib_2pt_unit_vec
    use lib_grid_mod,           only: lib_interp_lag_get_coef


    implicit none
    private

    public :: global_grid_gen_k2e

contains
    !===========================================================================
    !> @brief init kinked to extended grid remap coefficients
    subroutine global_grid_gen_k2e(gg)
        type(global_grid_type), intent(inout) :: gg
        !--- local
        integer :: i, j, k, n, ng, k2e_nord
        integer :: ii, jj, np
        integer :: isd, ied, jsd, jed, is, ie, js, je, khi, klo
        real(R_GRID), dimension(:,:,:), allocatable :: &
            ext_x, ext_y, kik_x, kik_y ! A-pt copies
        real(R_GRID), dimension(:), allocatable :: x, y ! x: kik(org), y: ext(tar)
        real(R_GRID) :: yy 

        !--- assign dims
        is = 1
        ie = gg%res
        js = is
        je = ie

        ng = gg%ng

        isd = is - ng
        ied = ie + ng
        jsd = isd
        jed = ied

        k2e_nord = gg%k2e_nord
        np = k2e_nord/2-1

        !--- allocate a-pt copies
        allocate( ext_x(isd:ied, jsd:jed, 6) )
        allocate( ext_y(isd:ied, jsd:jed, 6) )
        allocate( kik_x(isd:ied, jsd:jed, 6) )
        allocate( kik_y(isd:ied, jsd:jed, 6) )
        allocate( x    (is :ie ) )
        allocate( y    (isd:ied) )

        !--- assign a-pt copies
        do n = 1,6
            do j = jsd, jed
                do i = isd, ied
                    ! point
                    ii = i*2; jj = j*2
    
                    ext_x(i,j,n) = gg%ext_x(ii,jj,n)
                    ext_y(i,j,n) = gg%ext_y(ii,jj,n)
                    kik_x(i,j,n) = gg%kik_x(ii,jj,n)
                    kik_y(i,j,n) = gg%kik_y(ii,jj,n)
    
                enddo
            enddo
        enddo

        !--- get k2e_loc
        gg%k2e_loc(:,:,:) = -999
        gg%k2e_coef(:,:,:,:) = -999.
        do n = 1,6
            do ii = 1,ng

                !--- rmp s n
                j = js-ii
                call get_loc_x
                j = je+ii
                call get_loc_x

                !--- rmp w e
                i = is-ii
                call get_loc_y
                i = ie+ii
                call get_loc_y

            enddo
        enddo


        !--- deallocate a-pt copies
        deallocate( x )
        deallocate( y )

        deallocate( ext_x )
        deallocate( ext_y )
        deallocate( kik_x )
        deallocate( kik_y )

    contains
        !-----------------------------------------------------------------------
        subroutine get_loc_x
            x(:) = kik_x(is :ie,  j, n)
            y(:) = ext_x(isd:ied, j, n)

            do i = is-ii+1, ie+ii-1

                yy = y(i)
                klo = is
                khi = ie

                do while (khi-klo>1)
                    k = (khi+klo)/2
                    if (x(k) > yy) then
                        khi = k
                    else
                        klo = k
                    endif
                enddo
                klo = max(klo, is+np)
                klo = min(klo, ie-np-1)
                khi = klo+1

                gg%k2e_loc(i,j,n) = klo

                gg%k2e_coef(:,i,j,n) = lib_interp_lag_get_coef(yy,x(klo-np:khi+np))

            enddo

        end subroutine get_loc_x
        !-----------------------------------------------------------------------
        subroutine get_loc_y
            x(:) = kik_y(i, js :je,  n)
            y(:) = ext_y(i, jsd:jed, n)

            do j = js-ii+1, je+ii-1

                yy = y(j)
                klo = js
                khi = je

                do while (khi-klo>1)
                    k = (khi+klo)/2
                    if (x(k) > yy) then
                        khi = k
                    else
                        klo = k
                    endif
                enddo
                klo = max(klo, js+np)
                klo = min(klo, je-np-1)
                khi = klo+1

                gg%k2e_loc(i,j,n) = klo

                gg%k2e_coef(:,i,j,n) = lib_interp_lag_get_coef(yy,x(klo-np:khi+np))

            enddo

        end subroutine get_loc_y
        !-----------------------------------------------------------------------
    end subroutine global_grid_gen_k2e
    !===========================================================================
end module global_grid_gen_k2e_mod

