#ifndef CHTHOLLY_GET_GLOBAL_GRID_H
#define CHTHOLLY_GET_GLOBAL_GRID_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the global grid
 * 
 * This function initializes the global grid with the specified parameters.
 * 
 * @param res Grid resolution
 * @param ng Number of ghost cells
 * @param grid_type Type of grid
 * @param tile Tile number
 * @param isd Starting index in x direction (domain)
 * @param ied Ending index in x direction (domain)
 * @param jsd Starting index in y direction (domain)
 * @param jed Ending index in y direction (domain)
 */
void chtholly_global_grid_init(int res, int ng, int grid_type, int tile, 
                               int isd, int ied, int jsd, int jed);

/**
 * @brief Finalize the global grid
 * 
 * This function cleans up resources allocated by chtholly_global_grid_init()
 */
void chtholly_global_grid_end(void);

/**
 * @brief Get a_pt_ext array from global grid
 * 
 * @param pt Output array for a_pt_ext data
 */
void chtholly_global_grid_get_a_pt_ext(double *pt);

/**
 * @brief Get a_x_dg array from global grid
 * 
 * @param out_arr Output array for a_x_dg data
 */
void chtholly_global_grid_get_a_x_dg(double *out_arr);

/**
 * @brief Get a_y_dg array from global grid
 * 
 * @param out_arr Output array for a_y_dg data
 */
void chtholly_global_grid_get_a_y_dg(double *out_arr);

/**
 * @brief Get a_kik_x_dg array from global grid
 * 
 * @param out_arr Output array for a_kik_x_dg data
 */
void chtholly_global_grid_get_a_kik_x_dg(double *out_arr);

/**
 * @brief Get a_kik_y_dg array from global grid
 * 
 * @param out_arr Output array for a_kik_y_dg data
 */
void chtholly_global_grid_get_a_kik_y_dg(double *out_arr);

/**
 * @brief Get a_gco_dg array from global grid
 * 
 * @param out_arr Output array for a_gco_dg data
 */
void chtholly_global_grid_get_a_gco_dg(double *out_arr);

/**
 * @brief Get a_gct_dg array from global grid
 * 
 * @param out_arr Output array for a_gct_dg data
 */
void chtholly_global_grid_get_a_gct_dg(double *out_arr);

/**
 * @brief Get a_c2l_dg array from global grid
 * 
 * @param out_arr Output array for a_c2l_dg data
 */
void chtholly_global_grid_get_a_c2l_dg(double *out_arr);

/**
 * @brief Get a_l2c_dg array from global grid
 * 
 * @param out_arr Output array for a_l2c_dg data
 */
void chtholly_global_grid_get_a_l2c_dg(double *out_arr);

/**
 * @brief Get a_sina_dg array from global grid
 * 
 * @param out_arr Output array for a_sina_dg data
 */
void chtholly_global_grid_get_a_sina_dg(double *out_arr);

/**
 * @brief Get a_cosa_dg array from global grid
 * 
 * @param out_arr Output array for a_cosa_dg data
 */
void chtholly_global_grid_get_a_cosa_dg(double *out_arr);

/**
 * @brief Get a_dx_dg array from global grid
 * 
 * @param out_arr Output array for a_dx_dg data
 */
void chtholly_global_grid_get_a_dx_dg(double *out_arr);

/**
 * @brief Get a_dy_dg array from global grid
 * 
 * @param out_arr Output array for a_dy_dg data
 */
void chtholly_global_grid_get_a_dy_dg(double *out_arr);

/**
 * @brief Get a_da_dg array from global grid
 * 
 * @param out_arr Output array for a_da_dg data
 */
void chtholly_global_grid_get_a_da_dg(double *out_arr);

/**
 * @brief Get rda_dg array from global grid
 * 
 * @param out_arr Output array for rda_dg data
 */
void chtholly_global_grid_get_rda_dg(double *out_arr);

/**
 * @brief Get rdx_dg array from global grid
 * 
 * @param out_arr Output array for rdx_dg data
 */
void chtholly_global_grid_get_rdx_dg(double *out_arr);

/**
 * @brief Get rdy_dg array from global grid
 * 
 * @param out_arr Output array for rdy_dg data
 */
void chtholly_global_grid_get_rdy_dg(double *out_arr);

/**
 * @brief Get k2e_loc_dg array from global grid
 * 
 * @param out_arr Output array for k2e_loc_dg data
 */
void chtholly_global_grid_get_k2e_loc_dg(int *out_arr);

/**
 * @brief Get k2e_coef_dg array from global grid
 * 
 * @param out_arr Output array for k2e_coef_dg data
 */
void chtholly_global_grid_get_k2e_coef_dg(double *out_arr);

/**
 * @brief Get b_pt_dg array from global grid
 * 
 * @param out_arr Output array for b_pt_dg data
 */
void chtholly_global_grid_get_b_pt_dg(double *out_arr);

/**
 * @brief Get c_gco_dg array from global grid
 * 
 * @param out_arr Output array for c_gco_dg data
 */
void chtholly_global_grid_get_c_gco_dg(double *out_arr);

/**
 * @brief Get c_gct_dg array from global grid
 * 
 * @param out_arr Output array for c_gct_dg data
 */
void chtholly_global_grid_get_c_gct_dg(double *out_arr);

/**
 * @brief Get c_ct2ort_x_dg array from global grid
 * 
 * @param out_arr Output array for c_ct2ort_x_dg data
 */
void chtholly_global_grid_get_c_ct2ort_x_dg(double *out_arr);

/**
 * @brief Get c_ort2ct_x_dg array from global grid
 * 
 * @param out_arr Output array for c_ort2ct_x_dg data
 */
void chtholly_global_grid_get_c_ort2ct_x_dg(double *out_arr);

/**
 * @brief Get c_sina_dg array from global grid
 * 
 * @param out_arr Output array for c_sina_dg data
 */
void chtholly_global_grid_get_c_sina_dg(double *out_arr);

/**
 * @brief Get c_cosa_dg array from global grid
 * 
 * @param out_arr Output array for c_cosa_dg data
 */
void chtholly_global_grid_get_c_cosa_dg(double *out_arr);

/**
 * @brief Get c_dy_dg array from global grid
 * 
 * @param out_arr Output array for c_dy_dg data
 */
void chtholly_global_grid_get_c_dy_dg(double *out_arr);

/**
 * @brief Get d_gco_dg array from global grid
 * 
 * @param out_arr Output array for d_gco_dg data
 */
void chtholly_global_grid_get_d_gco_dg(double *out_arr);

/**
 * @brief Get d_gct_dg array from global grid
 * 
 * @param out_arr Output array for d_gct_dg data
 */
void chtholly_global_grid_get_d_gct_dg(double *out_arr);

/**
 * @brief Get d_ct2ort_y_dg array from global grid
 * 
 * @param out_arr Output array for d_ct2ort_y_dg data
 */
void chtholly_global_grid_get_d_ct2ort_y_dg(double *out_arr);

/**
 * @brief Get d_ort2ct_y_dg array from global grid
 * 
 * @param out_arr Output array for d_ort2ct_y_dg data
 */
void chtholly_global_grid_get_d_ort2ct_y_dg(double *out_arr);

/**
 * @brief Get d_sina_dg array from global grid
 * 
 * @param out_arr Output array for d_sina_dg data
 */
void chtholly_global_grid_get_d_sina_dg(double *out_arr);

/**
 * @brief Get d_cosa_dg array from global grid
 * 
 * @param out_arr Output array for d_cosa_dg data
 */
void chtholly_global_grid_get_d_cosa_dg(double *out_arr);

/**
 * @brief Get d_dx_dg array from global grid
 * 
 * @param out_arr Output array for d_dx_dg data
 */
void chtholly_global_grid_get_d_dx_dg(double *out_arr);

#ifdef __cplusplus
}
#endif

#endif /* CHTHOLLY_GET_GLOBAL_GRID_H */
