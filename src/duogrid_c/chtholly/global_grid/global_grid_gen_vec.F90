!-------------------------------------------------------------------------------
!> @brief generate vectors on the cubed-sphere
!> @author Xi.Chen <xic@princeton.edu>
!> @date 01/17/2020
!
!  REVISION HISTORY:
!  01/17/2020 - Initial Version
!-------------------------------------------------------------------------------

module global_grid_gen_vec_mod

    !------ fms modules
    use lib_grid_mod,           only: pi
    !------ AC modules
    use global_grid_data_mod,   only: global_grid_type
    use global_grid_util_mod,   only: global_grid_update_kik
    use lib_grid_mod,           only: R_GRID
    use lib_grid_mod,           only: lib_2pt_unit_vec

    implicit none
    private

    public :: global_grid_gen_eco
    public :: global_grid_gen_elonlat
    public :: global_grid_gen_mat

contains
    !===========================================================================
    !> @brief generate e1co and e2co on the cubed-sphere
    subroutine global_grid_gen_eco(gg)
        type(global_grid_type), intent(inout) :: gg
        !--- local
        integer :: i, j, k, nc, n, ng
        integer :: ii, jj
        integer :: isd, ied, jsd, jed, is, ie, js, je
        real(R_GRID), dimension(2) :: p1, p2

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

        !--- get ext eco
        do n = 1,6
            do j = jsd, jed
                do i = isd, ied

                    !--- e1co
                    if (i==ied) then
                        p1 = gg%pt_ext(:,i,j,n)
                        p2 = gg%pt_ext(:,i-1,j,n)
                        gg%ext_e1co(:,i,j,n) = - lib_2pt_unit_vec(p1,p2)
                    else
                        p1 = gg%pt_ext(:,i,j,n)
                        p2 = gg%pt_ext(:,i+1,j,n)
                        gg%ext_e1co(:,i,j,n) = lib_2pt_unit_vec(p1,p2)
                    endif

                    !--- e2co
                    if (j==jed) then
                        p1 = gg%pt_ext(:,i,j,n)
                        p2 = gg%pt_ext(:,i,j-1,n)
                        gg%ext_e2co(:,i,j,n) = - lib_2pt_unit_vec(p1,p2)
                    else
                        p1 = gg%pt_ext(:,i,j,n)
                        p2 = gg%pt_ext(:,i,j+1,n)
                        gg%ext_e2co(:,i,j,n) = lib_2pt_unit_vec(p1,p2)
                    endif

                enddo
            enddo
        enddo

        !--- get kik eco (a little more work)
        !------ default null
        gg%kik_e1co(:,:,:,:) = -999.
        gg%kik_e2co(:,:,:,:) = -999.

        !------ interior
        do n = 1,6
            do j = js,je
                do i = is,ie
                    gg%kik_e1co(:,i,j,n) = gg%ext_e1co(:,i,j,n)
                    gg%kik_e2co(:,i,j,n) = gg%ext_e2co(:,i,j,n)
                enddo
            enddo
        enddo

        !------ e1co
        do n = 1,6

            do j = js,je
                ! west
                do i = isd,is
                    if (i == is) then
                        p1 = gg%pt_kik(:,i,j,n)
                        p2 = gg%pt_kik(:,i-1,j,n)
                        gg%kik_e1co(:,i,j,n) = - lib_2pt_unit_vec(p1,p2)
                    else
                        p1 = gg%pt_kik(:,i,j,n)
                        p2 = gg%pt_kik(:,i+1,j,n)
                        gg%kik_e1co(:,i,j,n) = lib_2pt_unit_vec(p1,p2)
                    endif
                enddo
                ! east --- repeat the code to enhance readability
                do i = ie,ied
                    if (i == ied) then
                        p1 = gg%pt_kik(:,i,j,n)
                        p2 = gg%pt_kik(:,i-1,j,n)
                        gg%kik_e1co(:,i,j,n) = - lib_2pt_unit_vec(p1,p2)
                    else
                        p1 = gg%pt_kik(:,i,j,n)
                        p2 = gg%pt_kik(:,i+1,j,n)
                        gg%kik_e1co(:,i,j,n) = lib_2pt_unit_vec(p1,p2)
                    endif
                enddo

            enddo

            !--- south --- tile edge equals to ext
            do j = jsd,js-1
                do i = is,ie
                    if (i == ie) then
                        p1 = gg%pt_kik(:,i,j,n)
                        p2 = gg%pt_kik(:,i-1,j,n)
                        gg%kik_e1co(:,i,j,n) = - lib_2pt_unit_vec(p1,p2)
                    else
                        p1 = gg%pt_kik(:,i,j,n)
                        p2 = gg%pt_kik(:,i+1,j,n)
                        gg%kik_e1co(:,i,j,n) = lib_2pt_unit_vec(p1,p2)
                    endif
                enddo
            enddo
            !--- north
            do j = je+1,jed
                do i = is,ie
                    if (i == ie) then
                        p1 = gg%pt_kik(:,i,j,n)
                        p2 = gg%pt_kik(:,i-1,j,n)
                        gg%kik_e1co(:,i,j,n) = - lib_2pt_unit_vec(p1,p2)
                    else
                        p1 = gg%pt_kik(:,i,j,n)
                        p2 = gg%pt_kik(:,i+1,j,n)
                        gg%kik_e1co(:,i,j,n) = lib_2pt_unit_vec(p1,p2)
                    endif
                enddo
            enddo

        enddo ! n=1,6


        !------ e2co
        do n = 1,6

            do j = js,je
                ! west --- tile edge equals to ext
                do i = isd, is-1
                    if (j==je) then
                        p1 = gg%pt_kik(:,i,j,n)
                        p2 = gg%pt_kik(:,i,j-1,n)
                        gg%kik_e2co(:,i,j,n) = - lib_2pt_unit_vec(p1,p2)
                    else
                        p1 = gg%pt_kik(:,i,j,n)
                        p2 = gg%pt_kik(:,i,j+1,n)
                        gg%kik_e2co(:,i,j,n) = lib_2pt_unit_vec(p1,p2)
                    endif
                enddo
                ! east --- tile edge equals to ext
                do i = ie+1, ied 
                    if (j==je) then
                        p1 = gg%pt_kik(:,i,j,n)
                        p2 = gg%pt_kik(:,i,j-1,n)
                        gg%kik_e2co(:,i,j,n) = - lib_2pt_unit_vec(p1,p2)
                    else
                        p1 = gg%pt_kik(:,i,j,n)
                        p2 = gg%pt_kik(:,i,j+1,n)
                        gg%kik_e2co(:,i,j,n) = lib_2pt_unit_vec(p1,p2)
                    endif
                enddo
            enddo

            ! south
            do j = jsd, js
                do i = is,ie
                    if (j==js) then
                        p1 = gg%pt_kik(:,i,j,n)
                        p2 = gg%pt_kik(:,i,j-1,n)
                        gg%kik_e2co(:,i,j,n) = - lib_2pt_unit_vec(p1,p2)
                    else
                        p1 = gg%pt_kik(:,i,j,n)
                        p2 = gg%pt_kik(:,i,j+1,n)
                        gg%kik_e2co(:,i,j,n) = lib_2pt_unit_vec(p1,p2)
                    endif
                enddo
            enddo

            ! north
            do j = je, jed
                do i = is,ie
                    if (j==jed) then
                        p1 = gg%pt_kik(:,i,j,n)
                        p2 = gg%pt_kik(:,i,j-1,n)
                        gg%kik_e2co(:,i,j,n) = - lib_2pt_unit_vec(p1,p2)
                    else
                        p1 = gg%pt_kik(:,i,j,n)
                        p2 = gg%pt_kik(:,i,j+1,n)
                        gg%kik_e2co(:,i,j,n) = lib_2pt_unit_vec(p1,p2)
                    endif
                enddo
            enddo

        enddo ! n=1,6

        !--- nullify the tile corners (shared by different halos)
        do n = 1,6
            gg%kik_e1co(:,is,js,n) = -999.
            gg%kik_e1co(:,ie,js,n) = -999.
            gg%kik_e1co(:,ie,je,n) = -999.
            gg%kik_e1co(:,is,je,n) = -999.
            gg%kik_e2co(:,is,js,n) = -999.
            gg%kik_e2co(:,ie,js,n) = -999.
            gg%kik_e2co(:,ie,je,n) = -999.
            gg%kik_e2co(:,is,je,n) = -999.
        enddo


    end subroutine global_grid_gen_eco
    !===========================================================================
    !> @brief generate regular elon and elat on the cubed-sphere
    subroutine global_grid_gen_elonlat(gg)
        type(global_grid_type), intent(inout) :: gg
        !--- local
        integer :: i, j, k, nc, n, ng
        integer :: ii, jj
        integer :: isd, ied, jsd, jed, is, ie, js, je
        real(R_GRID), dimension(2) :: p1, p2
        real(R_GRID) :: lon, lat

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

        !--- get ext elonlat
        do n = 1,6
            do j = jsd, jed
                do i = isd, ied

                    lon = gg%pt_ext(1,i,j,n)
                    lat = gg%pt_ext(2,i,j,n)
                    gg%ext_elon(1,i,j,n) = -sin(lon)
                    gg%ext_elon(2,i,j,n) = cos(lon);
                    gg%ext_elon(3,i,j,n) = 0.
                    gg%ext_elat(1,i,j,n) = -sin(lat)*cos(lon);
                    gg%ext_elat(2,i,j,n) = -sin(lat)*sin(lon);
                    gg%ext_elat(3,i,j,n) = cos(lat)

                enddo
            enddo
        enddo

        !--- get kik elonlat
        !------ default null
        gg%kik_elon(:,:,:,:) = -999.
        gg%kik_elat(:,:,:,:) = -999.

        call global_grid_update_kik(gg%ext_elon, gg%kik_elon, isd, ied, ng, 3)
        call global_grid_update_kik(gg%ext_elat, gg%kik_elat, isd, ied, ng, 3)

    end subroutine global_grid_gen_elonlat
    !===========================================================================
    !> @brief generate sina cosa on the cubed-sphere
    subroutine global_grid_gen_mat(gg)
        type(global_grid_type), intent(inout) :: gg
        !--- local
        integer :: i, j, k, nc, n, ng
        integer :: ii, jj
        integer :: isd, ied, jsd, jed, is, ie, js, je
        real(R_GRID), dimension(2) :: p1, p2
        real(R_GRID) :: lon, lat, sina, cosa, coef
        logical, dimension(:,:), allocatable :: assigned
        real(R_GRID) :: gct11, gco12, gco22
        real(R_GRID) :: gct22, gco11, gco21
        real(R_GRID), dimension(3) :: e1, e2, elon, elat
        real(R_GRID), dimension(2,2) :: mat


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

        !--- allocate working array, and make corners true
        allocate(assigned(isd:ied, jsd:jed))
        assigned(:,:) = .true.
        assigned(is:ie, jsd:jed) = .false.
        assigned(isd:ied, js:je) = .false.
        assigned(is,js) = .true.
        assigned(is,je) = .true.
        assigned(ie,je) = .true.
        assigned(ie,js) = .true.

        !--- get ext sina and cosa
        do n = 1,6
            do j = jsd, jed
                do i = isd, ied

                    gg%ext_cosa(i,j,n) = dot_product( &
                        gg%ext_e1co(:,i,j,n), gg%ext_e2co(:,i,j,n))
                    gg%ext_sina(i,j,n) = sqrt(1. - &
                        gg%ext_cosa(i,j,n)*gg%ext_cosa(i,j,n))

                enddo
            enddo
        enddo

        !--- get kik elonlat
        !------ default cosa = 0, sina = 1
        gg%kik_cosa(:,:,:) = 0.
        gg%kik_sina(:,:,:) = 1.

        !------ interior
        do n = 1,6
            do j = js+1, je-1
                do i = is+1, ie-1

                    gg%kik_cosa(i,j,n) = gg%ext_cosa(i,j,n)
                    gg%kik_sina(i,j,n) = gg%ext_sina(i,j,n)
                    assigned(i,j) = .true.

                enddo
            enddo
        enddo

        !------ halo
        do n = 1,6
            do j = jsd, jed
                do i = isd, ied

                    if ( .not.assigned(i,j) ) then

                        gg%kik_cosa(i,j,n) = dot_product( &
                            gg%kik_e1co(:,i,j,n), gg%kik_e2co(:,i,j,n))
                        gg%kik_sina(i,j,n) = sqrt(1. - &
                            gg%kik_cosa(i,j,n)*gg%kik_cosa(i,j,n))

                        if (n==6) then
                            assigned(i,j) = .true.
                        endif

                    endif

                enddo
            enddo
        enddo

        !--- 2x2 matrix init
        assigned(:,:) = .false.

        do j = js,je
            do i = isd,ied
                assigned(i,j) = .true.
            enddo
        enddo

        do j = jsd,jed
            do i = is,ie
                assigned(i,j) = .true.
            enddo
        enddo

        do j = js+1,je-1
            do i = is+1,ie-1
                assigned(i,j) = .false.
            enddo
        enddo

        assigned(is,js) = .false.
        assigned(is,je) = .false.
        assigned(ie,je) = .false.
        assigned(ie,js) = .false.

        do n = 1,6
            do j = jsd, jed
                do i = isd, ied

                    sina = gg%ext_sina(i,j,n)
                    cosa = gg%ext_cosa(i,j,n)

                    ! ext g_co g_ctr
                    gg%ext_g_co(1,1,i,j,n) = 1.
                    gg%ext_g_co(1,2,i,j,n) = cosa
                    gg%ext_g_co(2,1,i,j,n) = cosa
                    gg%ext_g_co(2,2,i,j,n) = 1.

                    coef = 1./sina/sina
                    gg%ext_g_ctr(1,1,i,j,n) = coef
                    gg%ext_g_ctr(1,2,i,j,n) = -coef*cosa
                    gg%ext_g_ctr(2,1,i,j,n) = -coef*cosa
                    gg%ext_g_ctr(2,2,i,j,n) = coef

                    ! ext ct2ort_x
                    gct11 = gg%ext_g_ctr(1,1,i,j,n)
                    gco12 = gg%ext_g_co (1,2,i,j,n)
                    gco22 = gg%ext_g_co (2,2,i,j,n)
                    gct22 = gg%ext_g_ctr(2,2,i,j,n)
                    gco11 = gg%ext_g_co (1,1,i,j,n)
                    gco21 = gg%ext_g_co (2,1,i,j,n)

                    gg%ext_ct2ort_x(1,1,i,j,n) = 1./sqrt(gct11)
                    gg%ext_ct2ort_x(1,2,i,j,n) = 0.
                    gg%ext_ct2ort_x(2,1,i,j,n) = gco12/sqrt(gco22)
                    gg%ext_ct2ort_x(2,2,i,j,n) = sqrt(gco22)

                    ! ext ort2ct_x
                    gg%ext_ort2ct_x(1,1,i,j,n) = sqrt(gct11)
                    gg%ext_ort2ct_x(1,2,i,j,n) = 0.
                    gg%ext_ort2ct_x(2,1,i,j,n) = -gco12/gco22*sqrt(gct11)
                    gg%ext_ort2ct_x(2,2,i,j,n) = 1./sqrt(gco22)

                    ! ext ct2ort_y
                    gg%ext_ct2ort_y(1,1,i,j,n) = 0.
                    gg%ext_ct2ort_y(1,2,i,j,n) = 1./sqrt(gct22)
                    gg%ext_ct2ort_y(2,1,i,j,n) = sqrt(gco11)
                    gg%ext_ct2ort_y(2,2,i,j,n) = gco21/sqrt(gco11)

                    ! ext ort2ct_y
                    gg%ext_ort2ct_y(1,1,i,j,n) = -gco21/gco11*sqrt(gct22)
                    gg%ext_ort2ct_y(1,2,i,j,n) = 1./sqrt(gco11)
                    gg%ext_ort2ct_y(2,1,i,j,n) = sqrt(gct22)
                    gg%ext_ort2ct_y(2,2,i,j,n) = 0.

                    ! ext c2l
                    e1   = gg%ext_e1co(:,i,j,n)
                    e2   = gg%ext_e2co(:,i,j,n)
                    elon = gg%ext_elon(:,i,j,n)
                    elat = gg%ext_elat(:,i,j,n)
                    mat(1,1) = dot_product( e1, elon ) 
                    mat(1,2) = dot_product( e2, elon )
                    mat(2,1) = dot_product( e1, elat )
                    mat(2,2) = dot_product( e2, elat )

                    gg%ext_ct2rll(:,:,i,j,n) = mat(:,:)
 
                    coef = 1./(mat(1,1) * mat(2,2) - mat(1,2) * mat(2,1))

                    gg%ext_rll2ct(1,1,i,j,n) =  coef*mat(2,2)
                    gg%ext_rll2ct(1,2,i,j,n) = -coef*mat(1,2)
                    gg%ext_rll2ct(2,1,i,j,n) = -coef*mat(2,1)
                    gg%ext_rll2ct(2,2,i,j,n) =  coef*mat(1,1)

                    gg%ext_c2l(:,:,i,j,n) = matmul( &
                        gg%ext_ct2rll(:,:,i,j,n), gg%ext_g_ctr(:,:,i,j,n))
                    gg%ext_l2c(:,:,i,j,n) = matmul( &
                        gg%ext_g_co(:,:,i,j,n), gg%ext_rll2ct(:,:,i,j,n))

                enddo
            enddo
        enddo

        ! kik ct2rll
        assigned(:,:) = .false.

        do j = js,je
            do i = isd,ied
                assigned(i,j) = .true.
            enddo
        enddo

        do j = jsd,jed
            do i = is,ie
                assigned(i,j) = .true.
            enddo
        enddo

        do j = js,je
            do i = is,ie
                assigned(i,j) = .false.
            enddo
        enddo

        gg%kik_ct2rll(:,:,:,:,:) = -999.
        gg%kik_rll2ct(:,:,:,:,:) = -999.
        do n = 1,6
            do j = jsd, jed
                do i = isd, ied

                    if (assigned(i,j)) then
                        e1   = gg%kik_e1co(:,i,j,n)
                        e2   = gg%kik_e2co(:,i,j,n)
                        elon = gg%kik_elon(:,i,j,n)
                        elat = gg%kik_elat(:,i,j,n)
                        mat(1,1) = dot_product( e1, elon ) 
                        mat(1,2) = dot_product( e2, elon )
                        mat(2,1) = dot_product( e1, elat )
                        mat(2,2) = dot_product( e2, elat )

                        gg%kik_ct2rll(:,:,i,j,n) = mat(:,:)
 
                        coef = 1./(mat(1,1) * mat(2,2) - mat(1,2) * mat(2,1))

                        gg%kik_rll2ct(1,1,i,j,n) =  coef*mat(2,2)
                        gg%kik_rll2ct(1,2,i,j,n) = -coef*mat(1,2)
                        gg%kik_rll2ct(2,1,i,j,n) = -coef*mat(2,1)
                        gg%kik_rll2ct(2,2,i,j,n) =  coef*mat(1,1)

                    endif

                enddo
            enddo
        enddo

        assigned(:,:) = .false.

        do j = js,je
            do i = isd,ied
                assigned(i,j) = .true.
            enddo
        enddo

        do j = jsd,jed
            do i = is,ie
                assigned(i,j) = .true.
            enddo
        enddo

        do j = js+1,je-1
            do i = is+1,ie-1
                assigned(i,j) = .false.
            enddo
        enddo

        assigned(is,js) = .false.
        assigned(is,je) = .false.
        assigned(ie,je) = .false.
        assigned(ie,js) = .false.

        do n = 1,6
            do j = jsd, jed
                do i = isd, ied

                    if (assigned(i,j)) then


                        sina = gg%kik_sina(i,j,n)
                        cosa = gg%kik_cosa(i,j,n)

                        ! ext g_co g_ctr
                        gg%kik_g_co(1,1,i,j,n) = 1.
                        gg%kik_g_co(1,2,i,j,n) = cosa
                        gg%kik_g_co(2,1,i,j,n) = cosa
                        gg%kik_g_co(2,2,i,j,n) = 1.

                        coef = 1./sina/sina
                        gg%kik_g_ctr(1,1,i,j,n) = coef
                        gg%kik_g_ctr(1,2,i,j,n) = -coef*cosa
                        gg%kik_g_ctr(2,1,i,j,n) = -coef*cosa
                        gg%kik_g_ctr(2,2,i,j,n) = coef


                        gg%kik_c2l(:,:,i,j,n) = matmul( &
                            gg%kik_ct2rll(:,:,i,j,n), gg%kik_g_ctr(:,:,i,j,n))
                        gg%kik_l2c(:,:,i,j,n) = matmul( &
                            gg%kik_g_co(:,:,i,j,n), gg%kik_rll2ct(:,:,i,j,n))

                    endif


                enddo
            enddo
        enddo


        !--- deallocate working array
        deallocate( assigned )


    end subroutine global_grid_gen_mat
    !===========================================================================
end module global_grid_gen_vec_mod


