"""
File: remap.py
Description: Remapping functions for A-grid, C-grid and D-grid.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-20
Updated: 2025-09-20
"""

# Third-party imports
import jax
import jax.numpy as jnp

# Standard library imports
from typing import Tuple

# Local application imports
from operators.poly import vector_interpolation_ew, vector_interpolation_ns
from duogrid.duogrid import Duogrid as Dg

@jax.jit
def to_c_grid(u: jnp.ndarray, v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Remap the boundary of the C-grid.
    
    Args:
        u: velocity component [xsize, ysize]
        v: velocity component [xsize, ysize]
    
    Returns:
        tuple: (uc, vc) - C-grid remapped velocities
    """
    # Interpolate to obtain directional components
    ue, uw = vector_interpolation_ew(u)  # East, West components
    un, us = vector_interpolation_ns(v)  # North, South components

    xsize, ysize = Dg.mp.xsize, Dg.mp.ysize

    # Initialize output arrays
    uc = jnp.zeros_like(u)
    vc = jnp.zeros_like(v)

    # For uc in x-direction: average of east and west components
    uc = uc.at[3:xsize-2, :].set(0.5 * (ue[2:xsize-3, :] + uw[3:xsize-2, :]))
    
    # For vc in y-direction: average of north and south components
    vc = vc.at[:, 3:ysize-2].set(0.5 * (un[:, 2:ysize-3] + us[:, 3:ysize-2]))

    return uc, vc

@jax.jit
def to_d_grid(u: jnp.ndarray, v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Remap the boundary of the D-grid.
    
    Args:
        u: velocity component [xsize, ysize]
        v: velocity component [xsize, ysize]
    
    Returns:
        tuple: (ud, vd) - D-grid remapped velocities
    """
    # Interpolate to obtain directional components
    un, us = vector_interpolation_ns(u)  # North, South components
    ve, vw = vector_interpolation_ew(v)  # East, West components

    xsize, ysize = Dg.mp.xsize, Dg.mp.ysize

    # Initialize output arrays
    ud = jnp.zeros_like(u)
    vd = jnp.zeros_like(v)

    # For vd in x-direction: average of east and west components
    vd = vd.at[3:xsize-2, :].set(0.5 * (ve[2:xsize-3, :] + vw[3:xsize-2, :]))
    
    # For ud in y-direction: average of north and south components
    ud = ud.at[:, 3:ysize-2].set(0.5 * (un[:, 2:ysize-3] + us[:, 3:ysize-2]))

    return ud, vd

@jax.jit
def to_d_grid_upwind(u: jnp.ndarray, v: jnp.ndarray, 
                     uc: jnp.ndarray, vc: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Remap the boundary of the D-grid using upwind scheme.
    
    Args:
        u: velocity component [xsize, ysize]
        v: velocity component [xsize, ysize]
        uc: contravariant velocity for upwind selection (x-direction) [xsize, ysize]
        vc: contravariant velocity for upwind selection (y-direction) [xsize, ysize]
    
    Returns:
        tuple: (ud, vd) - D-grid remapped velocities with upwind scheme
    """
    # Interpolate to obtain directional components
    un, us = vector_interpolation_ns(u)  # North, South components
    ve, vw = vector_interpolation_ew(v)  # East, West components

    xsize, ysize = Dg.mp.xsize, Dg.mp.ysize

    # Initialize output arrays
    ud = jnp.zeros_like(u)
    vd = jnp.zeros_like(v)

    # Upwind scheme selection
    # For vd in x-direction: select upwind component based on sign of uc
    vd_upwind = jnp.where(
        uc[3:xsize-2, :] > 0.0,  # If uc > 0, select west (upwind) component
        ve[2:xsize-3, :],        # West component
        vw[3:xsize-2, :],        # East component
    )
    vd = vd.at[3:xsize-2, :].set(vd_upwind)

    # For ud in y-direction: select upwind component based on sign of vc
    ud_upwind = jnp.where(
        vc[:, 3:ysize-2] > 0.0,  # If vc > 0, select south (upwind) component
        un[:, 2:ysize-3],        # South component
        us[:, 3:ysize-2],        # North component
    )
    ud = ud.at[:, 3:ysize-2].set(ud_upwind)
    
    return ud, vd