"""
File: remap.py
Description: Remapping functions for A-grid, C-grid and D-grid.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-20
Updated: 2025-09-21 (add upwind scheme)
"""

# Third-party imports
import jax
import jax.numpy as jnp
import functools

# Standard library imports
from typing import Tuple

# Local application imports
from operators.poly import vector_interpolation_ew, vector_interpolation_ns
from duogrid import Dg
from mesh.communication import Communication


def to_a_grid(u: jnp.ndarray, v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Remap the boundary of the A-grid.
    
    Args:
        u: velocity component [xsize, ysize]
        v: velocity component [xsize, ysize]
    """
    uct = jnp.zeros_like(u).at[..., :].set(
        (Dg.a_gct[..., 0, 0] * u + Dg.a_gct[..., 0, 1] * v) * Dg.a_sina
    )
    vct = jnp.zeros_like(v).at[..., :].set(
        (Dg.a_gct[..., 1, 0] * u + Dg.a_gct[..., 1, 1] * v) * Dg.a_sina
    )

    return uct, vct

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

    # mean of eta(right,i+1) and eta(left,i-1)
    uc = jnp.zeros_like(u).at[..., 3:-2, :].set(
        0.5 * (ue[..., 2:-3, :] + uw[..., 3:-2, :])
        )
    vc = jnp.zeros_like(v).at[..., :, 3:-2].set(
        0.5 * (un[..., :, 2:-3] + us[..., :, 3:-2])
        )

    return uc, vc

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

    # mean of eta(right,j+1) and eta(left,j-1)
    ud = jnp.zeros_like(u).at[..., 3:-2, :].set(
        0.5 * (un[..., 2:-3, :] + us[..., 3:-2, :])
        )
    vd = jnp.zeros_like(v).at[..., :, 3:-2].set(
        0.5 * (ve[..., :, 2:-3] + vw[..., :, 3:-2])
        )

    return ud, vd

@functools.partial(jax.jit, static_argnums=())
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

    # Upwind scheme selection
    # For vd in x-direction: select upwind component based on sign of uc
    vd = jnp.zeros_like(v).at[..., 3:-2, :].set(
        jnp.where(
            uc[..., 3:-2, :] > 0.0,  # If uc > 0, select west (upwind) component
            ve[..., 2:-3, :],        # West component
            vw[..., 3:-2, :],        # East component
        )
    )
    # For ud in y-direction: select upwind component based on sign of vc
    ud = jnp.zeros_like(u).at[..., :, 3:-2].set(    
        jnp.where(
            vc[..., :, 3:-2] > 0.0,  # If vc > 0, select south (upwind) component
            un[..., :, 2:-3],        # South component
            us[..., :, 3:-2],        # North component
        )
    )

    return ud, vd

def vector_trans_2d(u: jnp.ndarray, v: jnp.ndarray) \
    -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Remap the boundary of the 2D vector.
    
    Args:
        u: velocity component [xsize, ysize]
        v: velocity component [xsize, ysize]

    Returns:
        tuple: (uct, vct, ub_cx, vb_cy, ub_cy, vb_cx) - 2D vector remapped velocities
    """

    # A-grid
    uct, vct = to_a_grid(u, v)

    # C-grid.communication
    ub_cx, vb_cy = to_c_grid(uct, vct)
    ub_cx, vb_cy = Communication.boundary_communication(ub_cx, vb_cy)

    # D-grid.upwind
    ub_cy, vb_cx = to_d_grid_upwind(u, v, ub_cx, vb_cy)

    return uct, vct, ub_cx, ub_cy, vb_cx, vb_cy