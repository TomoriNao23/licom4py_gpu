#ifndef CHTHOLLY_WRAPPER_H
#define CHTHOLLY_WRAPPER_H

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

#ifdef __cplusplus
}
#endif

#endif /* CHTHOLLY_WRAPPER_H */
