#ifndef CHTHOLLY_GET_DATE_H
#define CHTHOLLY_GET_DATE_H

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief Initialize the Chtholly FMS wrapper
 * 
 * This function initializes the FMS (Flexible Modeling System) framework
 * and sets up the cube domain configuration for parallel computation.
 * It must be called before any other Chtholly wrapper functions.
 * 
 * @note This function is thread-safe and should be called once per process
 */
void chtholly_init(void);

/**
 * @brief Finalize the Chtholly FMS wrapper
 * 
 * This function cleans up resources allocated by chtholly_init()
 * and properly shuts down the FMS framework.
 * 
 * @note This function should be called once at the end of the program
 *       to ensure proper cleanup of resources
 */
void chtholly_end(void);

/**
 * @brief Extend a 1D scalar field to a 2D field
 * 
 * This function extends a 1D scalar field to a 2D field using the current domain.
 * 
 * @param field The 1D scalar field to extend
 */
void chtholly_ext_scalar_2d(double *field);

/**
 * @brief Get the tile number from the Chtholly FMS wrapper.
 * 
 * This function gets the tile number from the Chtholly FMS wrapper.
 * 
 * @note This function is thread-safe and should be called once per process
 */
int chtholly_get_tile(void);

/**
 * @brief Get the xsize
 * 
 * This function gets the xsize from the Chtholly FMS wrapper.
 */
int chtholly_get_xsize(void);

/**
 * @brief Get the ysize
 * 
 * This function gets the ysize from the Chtholly FMS wrapper.
 */
int chtholly_get_ysize(void);

/**
 * @brief Get the mpp pe
 * 
 * This function gets the mpp pe from the Chtholly FMS wrapper.
 */
int chtholly_get_mpp_pe(void);

/**
 * @brief Get the is
 * 
 * This function gets the is from the Chtholly FMS wrapper.
 */
int chtholly_get_is(void);

/**
 * @brief Get the ie
 * 
 * This function gets the ie from the Chtholly FMS wrapper.
 */
int chtholly_get_ie(void);

/**
 * @brief Get the js
 * 
 * This function gets the js from the Chtholly FMS wrapper.
 */
int chtholly_get_js(void);

/**
 * @brief Get the je
 * 
 * This function gets the je from the Chtholly FMS wrapper.
 */
int chtholly_get_je(void);

/**
 * @brief Get the isd
 * 
 * This function gets the isd from the Chtholly FMS wrapper.
 */
int chtholly_get_isd(void);

/**
 * @brief Get the ied
 * 
 * This function gets the ied from the Chtholly FMS wrapper.
 */
int chtholly_get_ied(void);

/**
 * @brief Get the jsd
 * 
 * This function gets the jsd from the Chtholly FMS wrapper.
 */
int chtholly_get_jsd(void);

#ifdef __cplusplus
}
#endif

#endif /* CHTHOLLY_GET_DATE_H */