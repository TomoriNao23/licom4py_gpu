!-------------------------------------------------------------------------------
!> @brief global_grid_type for global_grid module
!> @author Xi.Chen <xic@princeton.edu>
!> @date 01/10/2020
!
!  REVISION HISTORY:
!  01/10/2020 - Initial Version
!-------------------------------------------------------------------------------

module global_grid_data_mod

    !------ fms modules
    !------ AC modules
    use lib_grid_mod,           only: R_GRID

    implicit none
    private

    type global_grid_type

        ! grid data

        !--- dim parameters
        integer :: grid_type = 0
        integer :: res=-999, ng=-999
        integer :: tile=-999
        integer :: isd=-999, ied=-999, jsd=-999, jed=-999
        
        real,dimension(2,6) :: west_pole=-1., north_pole=-1., tile_center=-1.

        real(kind=R_GRID),dimension(:,:,:,:),allocatable :: pt_ext, pt_kik
! chtholly
        real(kind=R_GRID),dimension(:,:,:),  allocatable :: a_pt_dg
        real(kind=R_GRID),dimension(:,:),  allocatable :: a_x_dg, a_y_dg
        real(kind=R_GRID),dimension(:,:),  allocatable :: a_kik_x_dg, a_kik_y_dg
        real(kind=R_GRID),dimension(:,:,:,:),  allocatable :: a_gco_dg, a_gct_dg
        real(kind=R_GRID),dimension(:,:,:,:),  allocatable :: a_c2l_dg, a_l2c_dg
        real(kind=R_GRID),dimension(:,:),  allocatable :: a_sina_dg, a_cosa_dg
        real(kind=R_GRID),dimension(:,:),  allocatable :: a_dx_dg, a_dy_dg
        real(kind=R_GRID),dimension(:,:),  allocatable :: a_da_dg
        real(kind=R_GRID),dimension(:,:),  allocatable :: rda_dg, rdx_dg, rdy_dg
        integer, dimension(:,:),  allocatable :: k2e_loc_dg
        real(kind=R_GRID),dimension(:,:,:),  allocatable :: k2e_coef_dg
        real(kind=R_GRID),dimension(:,:,:),  allocatable :: b_pt_dg
        real(kind=R_GRID),dimension(:,:,:,:),  allocatable :: c_gco_dg, c_gct_dg
        real(kind=R_GRID),dimension(:,:,:,:),  allocatable :: c_ct2ort_x_dg, c_ort2ct_x_dg
        real(kind=R_GRID),dimension(:,:),  allocatable :: c_sina_dg, c_cosa_dg
        real(kind=R_GRID),dimension(:,:),  allocatable :: c_dy_dg
        real(kind=R_GRID),dimension(:,:,:,:),  allocatable :: d_gco_dg, d_gct_dg
        real(kind=R_GRID),dimension(:,:,:,:),  allocatable :: d_ct2ort_y_dg, d_ort2ct_y_dg
        real(kind=R_GRID),dimension(:,:),  allocatable :: d_sina_dg, d_cosa_dg
        real(kind=R_GRID),dimension(:,:),  allocatable :: d_dx_dg
! end chtholly
        real(kind=R_GRID),dimension(:,:,:),  allocatable :: ext_x, kik_x
        real(kind=R_GRID),dimension(:,:,:),  allocatable :: ext_y, kik_y

        real(kind=R_GRID),dimension(:,:,:),  allocatable :: ext_dx, kik_dx
        real(kind=R_GRID),dimension(:,:,:),  allocatable :: ext_dy, kik_dy
        real(kind=R_GRID),dimension(:,:,:),  allocatable :: ext_da, kik_da

        real(kind=R_GRID),dimension(:,:,:,:),allocatable :: ext_e1co, kik_e1co
        real(kind=R_GRID),dimension(:,:,:,:),allocatable :: ext_e2co, kik_e2co
        real(kind=R_GRID),dimension(:,:,:,:),allocatable :: ext_elon, kik_elon
        real(kind=R_GRID),dimension(:,:,:,:),allocatable :: ext_elat, kik_elat

        real(kind=R_GRID),dimension(:,:,:),  allocatable :: ext_sina, kik_sina
        real(kind=R_GRID),dimension(:,:,:),  allocatable :: ext_cosa, kik_cosa

        real(kind=R_GRID),dimension(:,:,:,:,:),allocatable :: &
            ext_g_co, ext_g_ctr, &
            kik_g_co, kik_g_ctr, &
            ext_ct2ort_x, ext_ort2ct_x, &
            ext_ct2ort_y, ext_ort2ct_y, &
            ext_ct2rll, ext_rll2ct, &
            ext_c2l, ext_l2c, & ! co
            kik_ct2rll, kik_rll2ct, &
            kik_c2l, kik_l2c

        !--- k2e rmp parameter
        integer :: k2e_nord = 2
        integer, dimension(:,:,:), allocatable :: k2e_loc
        real(kind=R_GRID), dimension(:,:,:,:), allocatable :: k2e_coef

        !--- type status
        logical :: is_initialized = .false.

    end type global_grid_type

    public :: global_grid_type

end module global_grid_data_mod

