!-------------------------------------------------------------------------------
!> @brief base methods for global_grid module
!> @author Xi.Chen <xic@princeton.edu>
!> @date 01/10/2020
!
!  REVISION HISTORY:
!  01/10/2020 - Initial Version
!-------------------------------------------------------------------------------

module global_grid_base_mod

    !------ fms modules
    !------ AC modules
    use global_grid_data_mod,   only: global_grid_type

    use global_grid_alloc_mod,  only: global_grid_alloc, global_grid_dealloc
    use global_grid_gen_lonlat_mod, only: global_grid_gen_lonlat_equal_edge
    use global_grid_gen_coords_mod, only: global_grid_gen_coords
    use global_grid_gen_cell_mod,   only: global_grid_gen_ds
    use global_grid_gen_cell_mod,   only: global_grid_gen_da
    use global_grid_gen_vec_mod,    only: global_grid_gen_eco
    use global_grid_gen_vec_mod,    only: global_grid_gen_elonlat
    use global_grid_gen_vec_mod,    only: global_grid_gen_mat
    use global_grid_gen_k2e_mod,    only: global_grid_gen_k2e
    use constants_mod,      only: OMEGA,RADIUS

    implicit none
    private

    public :: global_grid_init
    public :: global_grid_end

contains
    !===========================================================================
    !> @brief init method for global_grid_type
    subroutine global_grid_init(gg, res, ng, grid_type,tile, isd, ied, jsd, jed)
        type(global_grid_type), intent(inout) :: gg
        integer, intent(in) :: res, ng, grid_type
        integer, intent(in) :: tile
        integer, intent(in) :: isd, ied, jsd, jed


        !--- check status
        if (gg%is_initialized) return

        !--- set dim parameters and grid_type
        gg%isd = isd
        gg%ied = ied
        gg%jsd = jsd
        gg%jed = jed
        gg%res  = res
        gg%ng   = ng
        gg%grid_type = grid_type
        gg%tile = tile

        !--- alloc
        call global_grid_alloc(gg)

        !--- init lonlat values
        call global_grid_gen_lonlat_equal_edge(gg)

        !--- init remap coords
        call global_grid_gen_coords(gg)

        !--- xic: do stretching here, need to update tropical belt rmp coords

        !--- init dx, dy
        call global_grid_gen_ds(gg)

        !--- init da
        call global_grid_gen_da(gg)

        !--- init vec
        call global_grid_gen_eco(gg)
        call global_grid_gen_elonlat(gg)
        call global_grid_gen_mat(gg)

        !--- init k2e rmp
        call global_grid_gen_k2e(gg)
        call global_grid_chtholly(gg, isd, ied, jsd, jed)

        !write(*,*) "res, ng, grid_type, tile, isd, ied, jsd, jed", &
        !            res, ng, grid_type, tile, isd, ied, jsd, jed, sum(gg%a_x_dg(:,:))

        !--- set status
        gg%is_initialized = .true.

    end subroutine global_grid_init
    !===========================================================================
    !> @brief end method for global_grid_type
    subroutine global_grid_end(gg)
        type(global_grid_type), intent(inout) :: gg

        !--- check status
        if (.not.gg%is_initialized) return


        !--- dealloc
        call global_grid_dealloc(gg)


        !--- set status
        gg%is_initialized = .false.

    end subroutine global_grid_end
    !===========================================================================
    subroutine global_grid_chtholly(gg, isd, ied, jsd, jed)
        type(global_grid_type), intent(inout) :: gg

        integer :: isd, ied, jsd, jed
        integer :: i, j, n, ii, jj
        real :: alpha, lon, lat, ubar
        real, dimension(2) :: u_rll, u_co2
        n = gg%tile

        do j = jsd, jed
        do i = isd, ied
        ii = i*2; jj = j*2
        gg%a_pt_dg(:,i,j) = gg%pt_ext(:,ii,jj,n)
        gg%a_x_dg(i,j) = gg%ext_x(ii,jj,n)
        gg%a_y_dg(i,j) = gg%ext_y(ii,jj,n)
        gg%a_kik_x_dg(i,j) = gg%kik_x(ii,jj,n)
        gg%a_kik_y_dg(i,j) = gg%kik_y(ii,jj,n)
        gg%a_gco_dg(:,:,i,j) = gg%ext_g_co(:,:,ii,jj,n)
        gg%a_gct_dg(:,:,i,j) = gg%ext_g_ctr(:,:,ii,jj,n)
        gg%a_c2l_dg(:,:,i,j) = gg%ext_c2l(:,:,ii,jj,n)
        gg%a_l2c_dg(:,:,i,j) = gg%ext_l2c(:,:,ii,jj,n)
        gg%a_sina_dg(i,j) = gg%ext_sina(ii,jj,n)
        gg%a_cosa_dg(i,j) = gg%ext_cosa(ii,jj,n)
        ii = i*2-1; jj = j*2
        gg%a_dx_dg(i,j) = gg%ext_dx(ii,jj,n)+gg%ext_dx(ii+1,jj,n)
        ii = i*2; jj = j*2-1
        gg%a_dy_dg(i,j) = gg%ext_dy(ii,jj,n)+gg%ext_dy(ii,jj+1,n)
        ii = i*2-1; jj = j*2-1
        gg%a_da_dg(i,j) = gg%ext_da(ii,jj,n)  +gg%ext_da(ii,jj+1,n)+ &
                           gg%ext_da(ii+1,jj,n)+gg%ext_da(ii+1,jj+1,n)
        gg%rda_dg(i,j)  = 1.0/gg%a_da_dg(i,j)
        gg%rdx_dg(i,j)  = 1.0/gg%a_dx_dg(i,j)
        gg%rdy_dg(i,j)  = 1.0/gg%a_dy_dg(i,j)
        gg%k2e_loc_dg(i,j) = gg%k2e_loc(i,j,n)
        gg%k2e_coef_dg(:,i,j) = gg%k2e_coef(:,i,j,n)
        end do
        end do
        
        do j = jsd, jed+1
        do i = isd, ied+1
        ii = i*2-1; jj = j*2-1
        gg%b_pt_dg(:,i,j) = gg%pt_ext(:,ii,jj,n)
        end do
        end do

        do j = jsd, jed
        do i = isd, ied+1
        ii = i*2-1; jj = j*2
        gg%c_gco_dg(:,:,i,j) = gg%ext_g_co(:,:,ii,jj,n)
        gg%c_gct_dg(:,:,i,j) = gg%ext_g_ctr(:,:,ii,jj,n)
        gg%c_ct2ort_x_dg(:,:,i,j) = gg%ext_ct2ort_x(:,:,ii,jj,n)
        gg%c_ort2ct_x_dg(:,:,i,j) = gg%ext_ort2ct_x(:,:,ii,jj,n)
        gg%c_sina_dg(i,j) = gg%ext_sina(ii,jj,n)
        gg%c_cosa_dg(i,j) = gg%ext_cosa(ii,jj,n)
        ii = i*2-1; jj = j*2-1
        gg%c_dy_dg(i,j) = gg%ext_dy(ii,jj,n)+gg%ext_dy(ii,jj+1,n)
        end do
        end do

        do j = jsd, jed+1
        do i = isd, ied
        ii = i*2; jj = j*2-1
        gg%d_gco_dg(:,:,i,j) = gg%ext_g_co(:,:,ii,jj,n)
        gg%d_gct_dg(:,:,i,j) = gg%ext_g_ctr(:,:,ii,jj,n)
        gg%d_ct2ort_y_dg(:,:,i,j) = gg%ext_ct2ort_y(:,:,ii,jj,n)
        gg%d_ort2ct_y_dg(:,:,i,j) = gg%ext_ort2ct_y(:,:,ii,jj,n)
        gg%d_sina_dg(i,j) = gg%ext_sina(ii,jj,n)
        gg%d_cosa_dg(i,j) = gg%ext_cosa(ii,jj,n)
        ii = i*2-1; jj = j*2-1
        gg%d_dx_dg(i,j) = gg%ext_dx(ii,jj,n)+gg%ext_dx(ii+1,jj,n)
        end do
        end do

        do j = jsd, jed
        do i = isd+1, ied
            ii = i*2-2; jj = j*2
            gg%c_dx_dg(i,j) = gg%ext_dx(ii,jj,n)+gg%ext_dx(ii+1,jj,n)
        enddo
        enddo

        do j = jsd+1, jed
        do i = isd, ied
            ii = i*2; jj = j*2-2
            gg%d_dy_dg(i,j) = gg%ext_dy(ii,jj,n)+gg%ext_dy(ii,jj+1,n)
        enddo
        enddo

        alpha = 0.0
        ubar = 1.0
         do j = jsd,jed
            do i = isd,ied
                ! ext
                lon = gg%a_pt_dg(1,i,j)
                lat = gg%a_pt_dg(2,i,j)
                !a_f
                gg%a_f_dg(i,j) = 2.*OMEGA*(-1.*cos(lon)*cos(lat)*sin(alpha) + &
                    sin(lat)*cos(alpha) )
                u_rll(1) = ubar * (cos(alpha)*cos(lat) + &
                    sin(alpha)*cos(lon)*sin(lat))
                u_rll(2) = -ubar * sin(alpha)*sin(lon)
                u_co2 = matmul(gg%a_l2c_dg(:,:,i,j), u_rll)
                ! ub, vb
                gg%ub(i,j) = u_co2(1)
                gg%vb(i,j) = u_co2(2)
                
            enddo
        enddo
       
    end subroutine global_grid_chtholly
end module global_grid_base_mod
    

