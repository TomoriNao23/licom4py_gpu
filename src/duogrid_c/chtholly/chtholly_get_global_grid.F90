module c_global_grid_mod

  use iso_c_binding
  use global_grid_mod, only : global_grid_type, global_grid_init, global_grid_end

  implicit none
  private

  public :: chtholly_global_grid_init
  public :: chtholly_global_grid_end
  public :: chtholly_global_grid_get_a_pt_ext
  public :: chtholly_global_grid_get_a_x_dg
  public :: chtholly_global_grid_get_a_y_dg
  public :: chtholly_global_grid_get_a_kik_x_dg
  public :: chtholly_global_grid_get_a_kik_y_dg
  public :: chtholly_global_grid_get_a_gco_dg
  public :: chtholly_global_grid_get_a_gct_dg
  public :: chtholly_global_grid_get_a_c2l_dg
  public :: chtholly_global_grid_get_a_l2c_dg
  public :: chtholly_global_grid_get_a_sina_dg
  public :: chtholly_global_grid_get_a_cosa_dg
  public :: chtholly_global_grid_get_a_dx_dg
  public :: chtholly_global_grid_get_a_dy_dg
  public :: chtholly_global_grid_get_a_da_dg
  public :: chtholly_global_grid_get_rda_dg
  public :: chtholly_global_grid_get_rdx_dg
  public :: chtholly_global_grid_get_rdy_dg
  public :: chtholly_global_grid_get_k2e_loc_dg
  public :: chtholly_global_grid_get_k2e_loc_i
  public :: chtholly_global_grid_get_k2e_loc_j
  public :: chtholly_global_grid_get_k2e_coef_dg
  public :: chtholly_global_grid_get_b_pt_dg
  public :: chtholly_global_grid_get_c_gco_dg
  public :: chtholly_global_grid_get_c_gct_dg
  public :: chtholly_global_grid_get_c_ct2ort_x_dg
  public :: chtholly_global_grid_get_c_ort2ct_x_dg
  public :: chtholly_global_grid_get_c_sina_dg
  public :: chtholly_global_grid_get_c_cosa_dg
  public :: chtholly_global_grid_get_c_dy_dg
  public :: chtholly_global_grid_get_d_gco_dg
  public :: chtholly_global_grid_get_d_gct_dg
  public :: chtholly_global_grid_get_d_ct2ort_y_dg
  public :: chtholly_global_grid_get_d_ort2ct_y_dg
  public :: chtholly_global_grid_get_d_sina_dg
  public :: chtholly_global_grid_get_d_cosa_dg
  public :: chtholly_global_grid_get_d_dx_dg

  public :: chtholly_global_grid_get_a_f_dg
  public :: chtholly_global_grid_get_c_dx_dg
  public :: chtholly_global_grid_get_d_dy_dg
  public :: chtholly_global_grid_get_ub
  public :: chtholly_global_grid_get_vb

  type(global_grid_type), save :: gg

contains

  subroutine chtholly_global_grid_init(res, ng, grid_type, tile, isd, ied, jsd, jed) bind(C, name="chtholly_global_grid_init")
    integer(c_int), value :: res
    integer(c_int), value :: ng
    integer(c_int), value :: grid_type
    integer(c_int), value :: tile
    integer(c_int), value :: isd
    integer(c_int), value :: ied
    integer(c_int), value :: jsd
    integer(c_int), value :: jed
    call global_grid_init(gg, int(res, kind=kind(gg%res)), &
    int(ng, kind=kind(gg%ng)), int(grid_type, &
    kind=kind(gg%grid_type)), int(tile, kind=kind(gg%tile)), &
    int(isd, kind=kind(gg%isd)), int(ied, kind=kind(gg%ied)), &
    int(jsd, kind=kind(gg%jsd)), int(jed, kind=kind(gg%jed)))
  end subroutine chtholly_global_grid_init

  subroutine chtholly_global_grid_end() bind(C, name="chtholly_global_grid_end")
    call global_grid_end(gg)
  end subroutine chtholly_global_grid_end

  subroutine chtholly_global_grid_get_a_pt_ext(pt) bind(C, name="chtholly_global_grid_get_a_pt_ext")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: pt(2, size(gg%a_pt_dg,2), size(gg%a_pt_dg,3))
    pt(:,:,:) = gg%a_pt_dg(:,:,:)
  end subroutine chtholly_global_grid_get_a_pt_ext

  subroutine chtholly_global_grid_get_a_x_dg(out_arr) bind(C, name="chtholly_global_grid_get_a_x_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%a_x_dg,1), size(gg%a_x_dg,2))
    out_arr(:,:) = gg%a_x_dg(:,:)
  end subroutine chtholly_global_grid_get_a_x_dg

  subroutine chtholly_global_grid_get_a_y_dg(out_arr) bind(C, name="chtholly_global_grid_get_a_y_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%a_y_dg,1), size(gg%a_y_dg,2))
    out_arr(:,:) = gg%a_y_dg(:,:)
  end subroutine chtholly_global_grid_get_a_y_dg

  subroutine chtholly_global_grid_get_a_kik_x_dg(out_arr) bind(C, name="chtholly_global_grid_get_a_kik_x_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%a_kik_x_dg,1), size(gg%a_kik_x_dg,2))
    out_arr(:,:) = gg%a_kik_x_dg(:,:)
  end subroutine chtholly_global_grid_get_a_kik_x_dg

  subroutine chtholly_global_grid_get_a_kik_y_dg(out_arr) bind(C, name="chtholly_global_grid_get_a_kik_y_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%a_kik_y_dg,1), size(gg%a_kik_y_dg,2))
    out_arr(:,:) = gg%a_kik_y_dg(:,:)
  end subroutine chtholly_global_grid_get_a_kik_y_dg

  subroutine chtholly_global_grid_get_a_gco_dg(out_arr) bind(C, name="chtholly_global_grid_get_a_gco_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(2,2, size(gg%a_gco_dg,3), size(gg%a_gco_dg,4))
    out_arr(:,:,:,:) = gg%a_gco_dg(:,:,:,:)
  end subroutine chtholly_global_grid_get_a_gco_dg

  subroutine chtholly_global_grid_get_a_gct_dg(out_arr) bind(C, name="chtholly_global_grid_get_a_gct_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(2,2, size(gg%a_gct_dg,3), size(gg%a_gct_dg,4))
    out_arr(:,:,:,:) = gg%a_gct_dg(:,:,:,:)
  end subroutine chtholly_global_grid_get_a_gct_dg

  subroutine chtholly_global_grid_get_a_c2l_dg(out_arr) bind(C, name="chtholly_global_grid_get_a_c2l_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(2,2, size(gg%a_c2l_dg,3), size(gg%a_c2l_dg,4))
    out_arr(:,:,:,:) = gg%a_c2l_dg(:,:,:,:)
  end subroutine chtholly_global_grid_get_a_c2l_dg

  subroutine chtholly_global_grid_get_a_l2c_dg(out_arr) bind(C, name="chtholly_global_grid_get_a_l2c_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(2,2, size(gg%a_l2c_dg,3), size(gg%a_l2c_dg,4))
    out_arr(:,:,:,:) = gg%a_l2c_dg(:,:,:,:)
  end subroutine chtholly_global_grid_get_a_l2c_dg

  subroutine chtholly_global_grid_get_a_sina_dg(out_arr) bind(C, name="chtholly_global_grid_get_a_sina_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%a_sina_dg,1), size(gg%a_sina_dg,2))
    out_arr(:,:) = gg%a_sina_dg(:,:)
  end subroutine chtholly_global_grid_get_a_sina_dg

  subroutine chtholly_global_grid_get_a_cosa_dg(out_arr) bind(C, name="chtholly_global_grid_get_a_cosa_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%a_cosa_dg,1), size(gg%a_cosa_dg,2))
    out_arr(:,:) = gg%a_cosa_dg(:,:)
  end subroutine chtholly_global_grid_get_a_cosa_dg

  subroutine chtholly_global_grid_get_a_dx_dg(out_arr) bind(C, name="chtholly_global_grid_get_a_dx_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%a_dx_dg,1), size(gg%a_dx_dg,2))
    out_arr(:,:) = gg%a_dx_dg(:,:)
  end subroutine chtholly_global_grid_get_a_dx_dg

  subroutine chtholly_global_grid_get_a_dy_dg(out_arr) bind(C, name="chtholly_global_grid_get_a_dy_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%a_dy_dg,1), size(gg%a_dy_dg,2))
    out_arr(:,:) = gg%a_dy_dg(:,:)
  end subroutine chtholly_global_grid_get_a_dy_dg

  subroutine chtholly_global_grid_get_a_da_dg(out_arr) bind(C, name="chtholly_global_grid_get_a_da_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%a_da_dg,1), size(gg%a_da_dg,2))
    out_arr(:,:) = gg%a_da_dg(:,:)
  end subroutine chtholly_global_grid_get_a_da_dg

  subroutine chtholly_global_grid_get_rda_dg(out_arr) bind(C, name="chtholly_global_grid_get_rda_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%rda_dg,1), size(gg%rda_dg,2))
    out_arr(:,:) = gg%rda_dg(:,:)
  end subroutine chtholly_global_grid_get_rda_dg

  subroutine chtholly_global_grid_get_rdx_dg(out_arr) bind(C, name="chtholly_global_grid_get_rdx_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%rdx_dg,1), size(gg%rdx_dg,2))
    out_arr(:,:) = gg%rdx_dg(:,:)
  end subroutine chtholly_global_grid_get_rdx_dg

  subroutine chtholly_global_grid_get_rdy_dg(out_arr) bind(C, name="chtholly_global_grid_get_rdy_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%rdy_dg,1), size(gg%rdy_dg,2))
    out_arr(:,:) = gg%rdy_dg(:,:)
  end subroutine chtholly_global_grid_get_rdy_dg

  subroutine chtholly_global_grid_get_k2e_loc_dg(out_arr) bind(C, name="chtholly_global_grid_get_k2e_loc_dg")
    use iso_c_binding, only: c_int
    integer(c_int), intent(out) :: out_arr(size(gg%k2e_loc_dg,1), size(gg%k2e_loc_dg,2))
    out_arr(:,:) = gg%k2e_loc_dg(:,:)
  end subroutine chtholly_global_grid_get_k2e_loc_dg

  subroutine chtholly_global_grid_get_k2e_loc_i(out_arr) bind(C, name="chtholly_global_grid_get_k2e_loc_i")
    use iso_c_binding, only: c_int
    integer(c_int), intent(out) :: out_arr(size(gg%k2e_loc_dg,1), size(gg%k2e_loc_dg,2))
    out_arr(:,:) = gg%k2e_loc_dg(:,:) - gg%isd
  end subroutine chtholly_global_grid_get_k2e_loc_i

  subroutine chtholly_global_grid_get_k2e_loc_j(out_arr) bind(C, name="chtholly_global_grid_get_k2e_loc_j")
    use iso_c_binding, only: c_int
    integer(c_int), intent(out) :: out_arr(size(gg%k2e_loc_dg,1), size(gg%k2e_loc_dg,2))
    out_arr(:,:) = gg%k2e_loc_dg(:,:) - gg%jsd
  end subroutine chtholly_global_grid_get_k2e_loc_j

  subroutine chtholly_global_grid_get_k2e_coef_dg(out_arr) bind(C, name="chtholly_global_grid_get_k2e_coef_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%k2e_coef_dg,1), size(gg%k2e_coef_dg,2), size(gg%k2e_coef_dg,3))
    out_arr(:,:,:) = gg%k2e_coef_dg(:,:,:)
  end subroutine chtholly_global_grid_get_k2e_coef_dg

  subroutine chtholly_global_grid_get_b_pt_dg(out_arr) bind(C, name="chtholly_global_grid_get_b_pt_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(2, size(gg%b_pt_dg,2), size(gg%b_pt_dg,3))
    out_arr(:,:,:) = gg%b_pt_dg(:,:,:)
  end subroutine chtholly_global_grid_get_b_pt_dg

  subroutine chtholly_global_grid_get_c_gco_dg(out_arr) bind(C, name="chtholly_global_grid_get_c_gco_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(2,2, size(gg%c_gco_dg,3), size(gg%c_gco_dg,4))
    out_arr(:,:,:,:) = gg%c_gco_dg(:,:,:,:)
  end subroutine chtholly_global_grid_get_c_gco_dg

  subroutine chtholly_global_grid_get_c_gct_dg(out_arr) bind(C, name="chtholly_global_grid_get_c_gct_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(2,2, size(gg%c_gct_dg,3), size(gg%c_gct_dg,4))
    out_arr(:,:,:,:) = gg%c_gct_dg(:,:,:,:)
  end subroutine chtholly_global_grid_get_c_gct_dg

  subroutine chtholly_global_grid_get_c_ct2ort_x_dg(out_arr) bind(C, name="chtholly_global_grid_get_c_ct2ort_x_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(2,2, size(gg%c_ct2ort_x_dg,3), size(gg%c_ct2ort_x_dg,4))
    out_arr(:,:,:,:) = gg%c_ct2ort_x_dg(:,:,:,:)
  end subroutine chtholly_global_grid_get_c_ct2ort_x_dg

  subroutine chtholly_global_grid_get_c_ort2ct_x_dg(out_arr) bind(C, name="chtholly_global_grid_get_c_ort2ct_x_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(2,2, size(gg%c_ort2ct_x_dg,3), size(gg%c_ort2ct_x_dg,4))
    out_arr(:,:,:,:) = gg%c_ort2ct_x_dg(:,:,:,:)
  end subroutine chtholly_global_grid_get_c_ort2ct_x_dg

  subroutine chtholly_global_grid_get_c_sina_dg(out_arr) bind(C, name="chtholly_global_grid_get_c_sina_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%c_sina_dg,1), size(gg%c_sina_dg,2))
    out_arr(:,:) = gg%c_sina_dg(:,:)
  end subroutine chtholly_global_grid_get_c_sina_dg

  subroutine chtholly_global_grid_get_c_cosa_dg(out_arr) bind(C, name="chtholly_global_grid_get_c_cosa_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%c_cosa_dg,1), size(gg%c_cosa_dg,2))
    out_arr(:,:) = gg%c_cosa_dg(:,:)
  end subroutine chtholly_global_grid_get_c_cosa_dg

  subroutine chtholly_global_grid_get_c_dy_dg(out_arr) bind(C, name="chtholly_global_grid_get_c_dy_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%c_dy_dg,1), size(gg%c_dy_dg,2))
    out_arr(:,:) = gg%c_dy_dg(:,:)
  end subroutine chtholly_global_grid_get_c_dy_dg

  subroutine chtholly_global_grid_get_d_gco_dg(out_arr) bind(C, name="chtholly_global_grid_get_d_gco_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(2,2, size(gg%d_gco_dg,3), size(gg%d_gco_dg,4))
    out_arr(:,:,:,:) = gg%d_gco_dg(:,:,:,:)
  end subroutine chtholly_global_grid_get_d_gco_dg

  subroutine chtholly_global_grid_get_d_gct_dg(out_arr) bind(C, name="chtholly_global_grid_get_d_gct_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(2,2, size(gg%d_gct_dg,3), size(gg%d_gct_dg,4))
    out_arr(:,:,:,:) = gg%d_gct_dg(:,:,:,:)
  end subroutine chtholly_global_grid_get_d_gct_dg

  subroutine chtholly_global_grid_get_d_ct2ort_y_dg(out_arr) bind(C, name="chtholly_global_grid_get_d_ct2ort_y_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(2,2, size(gg%d_ct2ort_y_dg,3), size(gg%d_ct2ort_y_dg,4))
    out_arr(:,:,:,:) = gg%d_ct2ort_y_dg(:,:,:,:)
  end subroutine chtholly_global_grid_get_d_ct2ort_y_dg

  subroutine chtholly_global_grid_get_d_ort2ct_y_dg(out_arr) bind(C, name="chtholly_global_grid_get_d_ort2ct_y_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(2,2, size(gg%d_ort2ct_y_dg,3), size(gg%d_ort2ct_y_dg,4))
    out_arr(:,:,:,:) = gg%d_ort2ct_y_dg(:,:,:,:)
  end subroutine chtholly_global_grid_get_d_ort2ct_y_dg

  subroutine chtholly_global_grid_get_d_sina_dg(out_arr) bind(C, name="chtholly_global_grid_get_d_sina_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%d_sina_dg,1), size(gg%d_sina_dg,2))
    out_arr(:,:) = gg%d_sina_dg(:,:)
  end subroutine chtholly_global_grid_get_d_sina_dg

  subroutine chtholly_global_grid_get_d_cosa_dg(out_arr) bind(C, name="chtholly_global_grid_get_d_cosa_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%d_cosa_dg,1), size(gg%d_cosa_dg,2))
    out_arr(:,:) = gg%d_cosa_dg(:,:)
  end subroutine chtholly_global_grid_get_d_cosa_dg

  subroutine chtholly_global_grid_get_d_dx_dg(out_arr) bind(C, name="chtholly_global_grid_get_d_dx_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%d_dx_dg,1), size(gg%d_dx_dg,2))
    out_arr(:,:) = gg%d_dx_dg(:,:)
  end subroutine chtholly_global_grid_get_d_dx_dg

  subroutine chtholly_global_grid_get_a_f_dg(out_arr) bind(C, name="chtholly_global_grid_get_a_f_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%a_f_dg,1), size(gg%a_f_dg,2))
    out_arr(:,:) = gg%a_f_dg(:,:)
  end subroutine chtholly_global_grid_get_a_f_dg

  subroutine chtholly_global_grid_get_c_dx_dg(out_arr) bind(C, name="chtholly_global_grid_get_c_dx_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%c_dx_dg,1), size(gg%c_dx_dg,2))
    out_arr(:,:) = gg%c_dx_dg(:,:)
  end subroutine chtholly_global_grid_get_c_dx_dg

  subroutine chtholly_global_grid_get_d_dy_dg(out_arr) bind(C, name="chtholly_global_grid_get_d_dy_dg")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%d_dy_dg,1), size(gg%d_dy_dg,2))
    out_arr(:,:) = gg%d_dy_dg(:,:)
  end subroutine chtholly_global_grid_get_d_dy_dg

  subroutine chtholly_global_grid_get_ub(out_arr) bind(C, name="chtholly_global_grid_get_ub")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%ub,1), size(gg%ub,2))
    out_arr(:,:) = gg%ub(:,:)
  end subroutine chtholly_global_grid_get_ub

  subroutine chtholly_global_grid_get_vb(out_arr) bind(C, name="chtholly_global_grid_get_vb")
    use iso_c_binding, only: c_double
    real(c_double), intent(out) :: out_arr(size(gg%vb,1), size(gg%vb,2))
    out_arr(:,:) = gg%vb(:,:)
  end subroutine chtholly_global_grid_get_vb

end module c_global_grid_mod