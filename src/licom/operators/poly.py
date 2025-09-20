"""
File: poly.py
Description: Polynomial interpolation functions.
    Specially (1) only using 3rd order polynomial interpolation.
              (2) no using "if" in the code.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-20 (only for vector, TODO: add scalar method)
Updated: 2025-09-20 
"""

# Third-party imports
import jax
import jax.numpy as jnp

# Standard library imports
from typing import Tuple

# Local application imports
from duogrid.duogrid import Duogrid as Dg

@jax.jit
def vector_interpolation_ew(eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vector Polynomial Interpolation for East-West direction (3rd order only)
    
    Args:
        eta: input field [xsize, ysize] - first dimension corresponds to isd:ied
    
    Returns:
        tuple: (eta_star_east, eta_star_west) - both of shape [xsize, ysize]
    """
    xsize= Dg.mp.xsize
    
    # 3rd order polynomial coefficients
    COEFF_3_E = jnp.array([1.0/30.0, -13.0/60.0, 47.0/60.0, 9.0/20.0, -1.0/20.0])  # East
    COEFF_3_W = jnp.array([-1.0/20.0, 9.0/20.0, 47.0/60.0, -13.0/60.0, 1.0/30.0])  # West
    
    # Create stacked slices for vectorized operation (East-West: i direction, first axis)
    eta_slices = jnp.stack([
        eta[0:xsize-4, :],    # i-2 terms
        eta[1:xsize-3, :],    # i-1 terms
        eta[2:xsize-2, :],    # i terms
        eta[3:xsize-1, :],    # i+1 terms
        eta[4:xsize, :]       # i+2 terms
    ], axis=0)
    
    # Efficient tensor contraction using einsum
    result_e = jnp.einsum('k,kij->ij', COEFF_3_E, eta_slices)
    result_w = jnp.einsum('k,kij->ij', COEFF_3_W, eta_slices)
    
    # Place results in output arrays (is-1:ie+1 corresponds to 2:xsize-2 in i direction)
    eta_star_e = jnp.zeros_like(eta).at[2:xsize-2, :].set(result_e)
    eta_star_w = jnp.zeros_like(eta).at[2:xsize-2, :].set(result_w)
    
    return eta_star_e, eta_star_w


@jax.jit
def vector_interpolation_ns(eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vector Polynomial Interpolation for North-South direction (3rd order only)
    
    Args:
        eta: input field [xsize, ysize] - second dimension corresponds to jsd:jed
    
    Returns:
        tuple: (eta_star_north, eta_star_south) - both of shape [xsize, ysize]
    """
    ysize = Dg.mp.ysize
    
    # 3rd order polynomial coefficients
    COEFF_3_N = jnp.array([1.0/30.0, -13.0/60.0, 47.0/60.0, 9.0/20.0, -1.0/20.0])  # North
    COEFF_3_S = jnp.array([-1.0/20.0, 9.0/20.0, 47.0/60.0, -13.0/60.0, 1.0/30.0])  # South
    
    # Create stacked slices for vectorized operation (North-South: j direction, second axis)
    eta_slices = jnp.stack([
        eta[:, 0:ysize-4],    # j-2 terms
        eta[:, 1:ysize-3],    # j-1 terms
        eta[:, 2:ysize-2],    # j terms
        eta[:, 3:ysize-1],    # j+1 terms
        eta[:, 4:ysize]       # j+2 terms
    ], axis=0)
    
    # Efficient tensor contraction using einsum
    result_n = jnp.einsum('k,kij->ij', COEFF_3_N, eta_slices)
    result_s = jnp.einsum('k,kij->ij', COEFF_3_S, eta_slices)
    
    # Place results in output arrays (js-1:je+1 corresponds to 2:ysize-2 in j direction)
    eta_star_n = jnp.zeros_like(eta).at[:, 2:ysize-2].set(result_n)
    eta_star_s = jnp.zeros_like(eta).at[:, 2:ysize-2].set(result_s)
    
    return eta_star_n, eta_star_s


# Batch processing function
@jax.jit
def batch_vector_interpolation(eta_batch: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Batch process multiple fields simultaneously (3rd order only)
    
    Args:
        eta_batch: input fields [batch_size, xsize, ysize]
    
    Returns:
        tuple: (eta_e_batch, eta_w_batch, eta_n_batch, eta_s_batch)
    """
    # Vectorized map over batch dimension
    eta_e_batch, eta_w_batch = jax.vmap(vector_interpolation_ew)(eta_batch)
    eta_n_batch, eta_s_batch = jax.vmap(vector_interpolation_ns)(eta_batch)
    
    return eta_e_batch, eta_w_batch, eta_n_batch, eta_s_batch