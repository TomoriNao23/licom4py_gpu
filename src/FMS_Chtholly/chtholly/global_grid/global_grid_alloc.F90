!-------------------------------------------------------------------------------
!> @brief allocate arrays in global_grid_type
!> @author Xi.Chen <xic@princeton.edu>
!> @date 01/10/2020
!
!  REVISION HISTORY:
!  01/10/2020 - Initial Version
!-------------------------------------------------------------------------------

module global_grid_alloc_mod

    !------ fms modules
    !------ AC modules
    use global_grid_data_mod,   only: global_grid_type

    implicit none
    private

    public :: global_grid_alloc
    public :: global_grid_dealloc

contains
    !===========================================================================
    subroutine global_grid_alloc(gg)
        type(global_grid_type), intent(inout) :: gg
        !--- local
        integer :: isd, ied, jsd, jed 

        !--- assign dims
        isd = 1 - 2*gg%ng
        ied = 2*gg%res + 1 + 2*gg%ng
        jsd = isd
        jed = ied

        !--- allocate arrays
        allocate( gg%pt_ext(2, isd:ied, jsd:jed, 6) )
        allocate( gg%pt_kik(2, isd:ied, jsd:jed, 6) )

        allocate( gg%ext_x(isd:ied, jsd:jed, 6) )
        allocate( gg%kik_x(isd:ied, jsd:jed, 6) )

        allocate( gg%ext_y(isd:ied, jsd:jed, 6) )
        allocate( gg%kik_y(isd:ied, jsd:jed, 6) )

        allocate( gg%ext_dx(isd:ied, jsd:jed, 6) )
        allocate( gg%kik_dx(isd:ied, jsd:jed, 6) )

        allocate( gg%ext_dy(isd:ied, jsd:jed, 6) )
        allocate( gg%kik_dy(isd:ied, jsd:jed, 6) )

        allocate( gg%ext_da(isd:ied, jsd:jed, 6) )
        allocate( gg%kik_da(isd:ied, jsd:jed, 6) )

        allocate( gg%ext_e1co(3, isd:ied, jsd:jed, 6) )
        allocate( gg%kik_e1co(3, isd:ied, jsd:jed, 6) )

        allocate( gg%ext_e2co(3, isd:ied, jsd:jed, 6) )
        allocate( gg%kik_e2co(3, isd:ied, jsd:jed, 6) )

        allocate( gg%ext_elon(3, isd:ied, jsd:jed, 6) )
        allocate( gg%kik_elon(3, isd:ied, jsd:jed, 6) )

        allocate( gg%ext_elat(3, isd:ied, jsd:jed, 6) )
        allocate( gg%kik_elat(3, isd:ied, jsd:jed, 6) )

        allocate( gg%ext_sina(isd:ied, jsd:jed, 6) )
        allocate( gg%kik_sina(isd:ied, jsd:jed, 6) )

        allocate( gg%ext_cosa(isd:ied, jsd:jed, 6) )
        allocate( gg%kik_cosa(isd:ied, jsd:jed, 6) )

        allocate( gg%ext_g_co (2, 2, isd:ied, jsd:jed, 6) )
        allocate( gg%ext_g_ctr(2, 2, isd:ied, jsd:jed, 6) )

        allocate( gg%kik_g_co (2, 2, isd:ied, jsd:jed, 6) )
        allocate( gg%kik_g_ctr(2, 2, isd:ied, jsd:jed, 6) )

        allocate( gg%ext_ct2ort_x(2, 2, isd:ied, jsd:jed, 6) )
        allocate( gg%ext_ort2ct_x(2, 2, isd:ied, jsd:jed, 6) )
        allocate( gg%ext_ct2ort_y(2, 2, isd:ied, jsd:jed, 6) )
        allocate( gg%ext_ort2ct_y(2, 2, isd:ied, jsd:jed, 6) )
        allocate( gg%ext_ct2rll  (2, 2, isd:ied, jsd:jed, 6) )
        allocate( gg%ext_rll2ct  (2, 2, isd:ied, jsd:jed, 6) )
        allocate( gg%kik_ct2rll  (2, 2, isd:ied, jsd:jed, 6) )
        allocate( gg%kik_rll2ct  (2, 2, isd:ied, jsd:jed, 6) )
        allocate( gg%ext_c2l     (2, 2, isd:ied, jsd:jed, 6) )
        allocate( gg%ext_l2c     (2, 2, isd:ied, jsd:jed, 6) )
        allocate( gg%kik_c2l     (2, 2, isd:ied, jsd:jed, 6) )
        allocate( gg%kik_l2c     (2, 2, isd:ied, jsd:jed, 6) )

        !--- reassign a-pt dims for k2e
        isd = 1 - gg%ng
        ied = gg%res + gg%ng
        jsd = isd
        jed = ied

        allocate( gg%k2e_loc              (isd:ied, jsd:jed, 6) )
        allocate( gg%k2e_coef(gg%k2e_nord, isd:ied, jsd:jed, 6) )

        isd = gg%isd
        ied = gg%ied
        jsd = gg%jsd
        jed = gg%jed

        allocate( gg%a_pt_dg(2, isd:ied,  jsd:jed  ) )
        allocate( gg%a_x_dg    (isd:ied, jsd:jed ) )
        allocate( gg%a_y_dg    (isd:ied, jsd:jed ) )
        allocate( gg%a_kik_x_dg(isd:ied, jsd:jed ) )
        allocate( gg%a_kik_y_dg(isd:ied, jsd:jed ) )
        allocate( gg%a_gco_dg(2, 2, isd:ied, jsd:jed ) )
        allocate( gg%a_gct_dg(2, 2, isd:ied, jsd:jed ) )
        allocate( gg%a_c2l_dg(2, 2, isd:ied, jsd:jed ) )
        allocate( gg%a_l2c_dg(2, 2, isd:ied, jsd:jed ) )
        allocate( gg%a_sina_dg(isd:ied, jsd:jed ) )
        allocate( gg%a_cosa_dg(isd:ied, jsd:jed ) )
        allocate( gg%a_da_dg(isd:ied, jsd:jed ) )
        allocate( gg%a_dx_dg(isd:ied, jsd:jed ) )
        allocate( gg%a_dy_dg(isd:ied, jsd:jed ) )
        allocate( gg%rda_dg(isd:ied, jsd:jed ) )
        allocate( gg%rdx_dg(isd:ied, jsd:jed ) )
        allocate( gg%rdy_dg(isd:ied, jsd:jed ) )
        allocate( gg%k2e_loc_dg(isd:ied, jsd:jed ) )
        allocate( gg%k2e_coef_dg(2, isd:ied, jsd:jed ) )

        allocate(gg%b_pt_dg(2, isd:ied+1, jsd:jed+1 ) )
        allocate(gg%c_gco_dg(2, 2, isd:ied+1, jsd:jed ) )
        allocate(gg%c_gct_dg(2, 2, isd:ied+1, jsd:jed ) )
        allocate(gg%c_ct2ort_x_dg(2, 2, isd:ied+1, jsd:jed ) )
        allocate(gg%c_ort2ct_x_dg(2, 2, isd:ied+1, jsd:jed ) )
        allocate(gg%c_sina_dg(isd:ied+1, jsd:jed ) )
        allocate(gg%c_cosa_dg(isd:ied+1, jsd:jed ) )
        allocate(gg%c_dy_dg(isd:ied+1, jsd:jed ) )
        allocate(gg%d_gco_dg(2, 2, isd:ied, jsd:jed+1 ) )
        allocate(gg%d_gct_dg(2, 2, isd:ied, jsd:jed+1 ) )
        allocate(gg%d_ct2ort_y_dg(2, 2, isd:ied, jsd:jed+1 ) )
        allocate(gg%d_ort2ct_y_dg(2, 2, isd:ied, jsd:jed+1 ) )
        allocate(gg%d_sina_dg(isd:ied, jsd:jed+1 ) )
        allocate(gg%d_cosa_dg(isd:ied, jsd:jed+1 ) )
        allocate(gg%d_dx_dg(isd:ied, jsd:jed+1 ) )


    end subroutine global_grid_alloc
    !===========================================================================
    subroutine global_grid_dealloc(gg)
        type(global_grid_type), intent(inout) :: gg

        !--- deallocate arrays
        deallocate( gg%k2e_loc  )
        deallocate( gg%k2e_coef )

        deallocate( gg%ext_ct2ort_x )
        deallocate( gg%ext_ort2ct_x )
        deallocate( gg%ext_ct2ort_y )
        deallocate( gg%ext_ort2ct_y )
        deallocate( gg%ext_ct2rll   )
        deallocate( gg%ext_rll2ct   )
        deallocate( gg%kik_ct2rll   )
        deallocate( gg%kik_rll2ct   )
        deallocate( gg%ext_c2l      )
        deallocate( gg%ext_l2c      )
        deallocate( gg%kik_c2l      )
        deallocate( gg%kik_l2c      )


        deallocate( gg%ext_g_co  )
        deallocate( gg%ext_g_ctr )
        deallocate( gg%kik_g_co  )
        deallocate( gg%kik_g_ctr )


        deallocate( gg%ext_sina )
        deallocate( gg%kik_sina )

        deallocate( gg%ext_cosa )
        deallocate( gg%kik_cosa )

        deallocate( gg%ext_elon )
        deallocate( gg%kik_elon )

        deallocate( gg%ext_elat )
        deallocate( gg%kik_elat )

        deallocate( gg%ext_e1co )
        deallocate( gg%kik_e1co )

        deallocate( gg%ext_e2co )
        deallocate( gg%kik_e2co )

        deallocate( gg%ext_dx )
        deallocate( gg%kik_dx )

        deallocate( gg%ext_dy )
        deallocate( gg%kik_dy )

        deallocate( gg%ext_da )
        deallocate( gg%kik_da )

        deallocate( gg%ext_x )
        deallocate( gg%kik_x )

        deallocate( gg%ext_y )
        deallocate( gg%kik_y )

        deallocate( gg%pt_ext )
        deallocate( gg%pt_kik )

        deallocate( gg%a_pt_dg )
        deallocate( gg%a_x_dg )
        deallocate( gg%a_y_dg )
        deallocate( gg%a_kik_x_dg )
        deallocate( gg%a_kik_y_dg )
        deallocate( gg%a_gco_dg )
        deallocate( gg%a_gct_dg )
        deallocate( gg%a_c2l_dg )
        deallocate( gg%a_l2c_dg )
        deallocate( gg%a_sina_dg )
        deallocate( gg%a_cosa_dg )
        deallocate( gg%a_da_dg )
        deallocate( gg%a_dx_dg )
        deallocate( gg%a_dy_dg )
        deallocate( gg%rda_dg )
        deallocate( gg%rdx_dg )
        deallocate( gg%rdy_dg )
        deallocate( gg%k2e_loc_dg )
        deallocate( gg%k2e_coef_dg )
        deallocate( gg%b_pt_dg )
        deallocate( gg%c_gco_dg )
        deallocate( gg%c_gct_dg )
        deallocate( gg%c_ct2ort_x_dg )
        deallocate( gg%c_ort2ct_x_dg )
        deallocate( gg%c_sina_dg )
        deallocate( gg%c_cosa_dg )
        deallocate( gg%c_dy_dg )
        deallocate( gg%d_gco_dg )
        deallocate( gg%d_gct_dg )
        deallocate( gg%d_ct2ort_y_dg )
        deallocate( gg%d_ort2ct_y_dg )
        deallocate( gg%d_sina_dg )
        deallocate( gg%d_cosa_dg )
        deallocate( gg%d_dx_dg )

    end subroutine global_grid_dealloc
    !===========================================================================
end module global_grid_alloc_mod

