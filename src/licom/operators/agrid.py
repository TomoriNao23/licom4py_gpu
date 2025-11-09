"""
File: agrid.py
Description: A-grid operators.
include vorticity, divergence, gradient.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-21
Updated: 2025-09-21
"""

# Third-party imports
import jax
import jax.numpy as jnp
import functools

# Local application imports
from duogrid.duogrid import Duogrid as Dg
from operators.poly import scalar_interpolation_x, scalar_interpolation_y


@functools.partial(jax.jit, static_argnums=())
def agrid_vorticity(vv: jnp.ndarray, uu: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the vorticity on the A-grid.

    Args:
        uu: u velocity component [xsize, ysize]
        vv: v velocity component [xsize, ysize]

    Formula:
        vort(i,j) = rda(i,j) * (
            uu(i,j) * d_dx(i,j) +
            vv(i+1,j) * c_dy(i+1,j) -
            uu(i,j+1) * d_dx(i,j+1) +
            vv(i,j) * c_dy(i,j)
        )

    Returns:
        vort: vorticity [xsize, ysize]
    """

    vort = jnp.zeros_like(uu).at[:-1, :-1].set(
        Dg.rda[:-1, :-1] * (
            uu[:-1, :-1] * Dg.d_dx[:-1, :-2] +
            vv[1:, :-1] * Dg.c_dy[1:-1, :-1] -        # vv(i+1,j) * c_dy(i+1,j) 
            uu[:-1, 1:] * Dg.d_dx[:-1, 1:-1] -        # uu(i,j+1) * d_dx(i,j+1)
            vv[:-1, :-1] * Dg.c_dy[:-2, :-1]          # vv(i,j) * c_dy(i,j)
        )
    )
    
    return vort

@functools.partial(jax.jit, static_argnums=())
def agrid_div(uu: jnp.ndarray, vv: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the divergence on the A-grid.

    Args:
        uu: u velocity component [xsize, ysize]
        vv: v velocity component [xsize, ysize]

    Returns:
        div: divergence [xsize, ysize]
    """
    div = jnp.zeros_like(uu).at[:-1, :-1].set(
        Dg.rda[:-1, :-1] * (
            uu[1:, :-1] * Dg.c_dy[1:-1, :-1] -        # UX(i+1,j) * c_dy(i+1,j)
            uu[:-1, :-1] * Dg.c_dy[:-2, :-1] +        # UX(i,j) * c_dy(i,j)
            vv[:-1, 1:] * Dg.d_dx[:-1, 1:-1] -        # UY(i,j+1) * d_dx(i,j+1)  
            vv[:-1, :-1] * Dg.d_dx[:-1, :-2]          # UY(i,j) * d_dx(i,j)
        )
    )

    return div

@functools.partial(jax.jit, static_argnums=())
def agrid_grad(eta: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the gradient on the A-grid.

    Args:
        eta: scalar field [xsize, ysize]
    Returns:
        gradx: gradient in x direction [xsize, ysize]
        grady: gradient in y direction [xsize, ysize]
    """

    uhalf = scalar_interpolation_x(eta)
    vhalf = scalar_interpolation_y(eta)

    gradx = jnp.zeros_like(eta).at[3:-3,3:-3].set(
        Dg.rdx[3:-3,3:-3] * (
            uhalf[4:-2,3:-3] - uhalf[3:-3,3:-3]
        )
    )
    grady = jnp.zeros_like(eta).at[3:-3,3:-3].set(
        Dg.rdy[3:-3,3:-3] * (
            vhalf[3:-3,4:-2] - vhalf[3:-3,3:-3]
        )
    )

    return gradx, grady