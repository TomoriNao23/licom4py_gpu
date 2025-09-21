"""
File: poly.py
Description: Polynomial interpolation functions.
    Specially (1) only using 3rd order polynomial interpolation.
              (2) no using "if" in the code.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-20 (only for vector, TODO: add scalar method)
Updated: 2025-09-21 (add scalar method, TODO: topography)
"""

# Third-party imports
import functools
import jax
import jax.numpy as jnp

# Standard library imports
from typing import Tuple

# 3rd order polynomial coefficients
Ep23vm = -1.0/20.0
Ep13vm = 9.0/20.0
Ep03vm = 47.0/60.0
Em13vm = -13.0/60.0
Em23vm = 1.0/30.0

M13vm = 37.0/60.0
M23vm = -2.0/15.0
M33vm = 1.0/60.0

@functools.partial(jax.jit, static_argnums=())
def vector_interpolation_ew(eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vector Polynomial Interpolation for East-West direction (3rd order only)
    
    Args:
        eta: input field [xsize, ysize] - first dimension corresponds to isd:ied

    Formula:
        eta_star_e(i,j) = Em23vm * eta(i-2,j) + 
                          Em13vm * eta(i-1,j) + 
                          Ep03vm * eta(i,j) + 
                          Ep13vm * eta(i+1,j) + 
                          Ep23vm * eta(i+2,j)
        eta_star_w(i,j) = Ep23vm * eta(i-2,j) + 
                          Ep13vm * eta(i-1,j) + 
                          Ep03vm * eta(i,j) + 
                          Em13vm * eta(i+1,j) + 
                          Em23vm * eta(i+2,j)
    
    Returns:
        tuple: (eta_star_east, eta_star_west) - both of shape [xsize, ysize]
    """
    
    eta_star_e = jnp.zeros_like(eta).at[2:-2, :].set(
        Em23vm * eta[0:-4, :] + 
        Em13vm * eta[1:-3, :] + 
        Ep03vm * eta[2:-2, :] + 
        Ep13vm * eta[3:-1, :] + 
        Ep23vm * eta[4:, :]
    )
    
    eta_star_w = jnp.zeros_like(eta).at[2:-2, :].set(
        Ep23vm * eta[0:-4, :] + 
        Ep13vm * eta[1:-3, :] + 
        Ep03vm * eta[2:-2, :] + 
        Em13vm * eta[3:-1, :] + 
        Em23vm * eta[4:, :]
    )
    
    return eta_star_e, eta_star_w


@functools.partial(jax.jit, static_argnums=())
def vector_interpolation_ns(eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vector Polynomial Interpolation for North-South direction (3rd order only)
    
    Args:
        eta: input field [xsize, ysize] - second dimension corresponds to jsd:jed

    Formula:
        eta_star_n(i,j) = Em23vm * eta(i,j-2) + 
                          Em13vm * eta(i,j-1) + 
                          Ep03vm * eta(i,j) + 
                          Ep13vm * eta(i,j+1) + 
                          Ep23vm * eta(i,j+2)
        eta_star_s(i,j) = Ep23vm * eta(i,j-2) + 
                          Ep13vm * eta(i,j-1) + 
                          Ep03vm * eta(i,j) + 
                          Em13vm * eta(i,j+1) + 
                          Em23vm * eta(i,j+2)

    Returns:
        tuple: (eta_star_north, eta_star_south) - both of shape [xsize, ysize]
    """
    eta_star_n = jnp.zeros_like(eta).at[:, 2:-2].set(
        Em23vm * eta[:, 0:-4] + 
        Em13vm * eta[:, 1:-3] + 
        Ep03vm * eta[:, 2:-2] + 
        Ep13vm * eta[:, 3:-1] + 
        Ep23vm * eta[:, 4:]
    )
    
    eta_star_s = jnp.zeros_like(eta).at[:, 2:-2].set(
        Ep23vm * eta[:, 0:-4] + 
        Ep13vm * eta[:, 1:-3] + 
        Ep03vm * eta[:, 2:-2] + 
        Em13vm * eta[:, 3:-1] + 
        Em23vm * eta[:, 4:]
    )
    
    return eta_star_n, eta_star_s

# processing function
@functools.partial(jax.jit, static_argnums=())
def vector_interpolation(eta: jnp.ndarray) \
    -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Vector Polynomial Interpolation (3rd order only)
    
    Args:
        eta: input field [xsize, ysize]
    
    Returns:
        tuple: (eta_e, eta_w, eta_n, eta_s) - all of shape [xsize, ysize]
    """
    eta_e, eta_w = vector_interpolation_ew(eta)
    eta_n, eta_s = vector_interpolation_ns(eta)
    
    return eta_e, eta_w, eta_n, eta_s

@functools.partial(jax.jit, static_argnums=())
def scalar_interpolation_x(scal: jnp.ndarray) -> jnp.ndarray:
    """
    Scalar Polynomial Interpolation for x direction (3rd order)
    Based on Fortran code with coefficients M13vm, M23vm, M33vm
    
    Args:
        scal: input scalar field [xsize, ysize]

    Formula:
        scal_out(i,j) = M13vm * (scal(i-1,j) + scal(i,j)) + 
                        M23vm * (scal(i-2,j) + scal(i+1,j)) + 
                        M33vm * (scal(i-3,j) + scal(i+2,j))

    Returns:
        scal_out: interpolated scalar field [xsize, ysize]
    """
    
    scal_out = jnp.zeros_like(scal).at[3:-2, 3:-2].set(
        (
            M13vm * (scal[2:-3, 3:-2] + scal[3:-2, 3:-2]) + 
            M23vm * (scal[1:-4, 3:-2] + scal[4:-1, 3:-2]) + 
            M33vm * (scal[:-5, 3:-2] + scal[5:, 3:-2])
        )
    )
    
    return scal_out


@functools.partial(jax.jit, static_argnums=())
def scalar_interpolation_y(scal: jnp.ndarray) -> jnp.ndarray:
    """
    Scalar Polynomial Interpolation for y direction (3rd order)
    Based on Fortran code with coefficients M13vm, M23vm, M33vm
    
    Args:
        scal: input scalar field [xsize, ysize]
    
    Formula:
        scal_out(i,j) = M13vm * (scal(i,j-1) + scal(i,j)) + 
                        M23vm * (scal(i,j-2) + scal(i,j+1)) + 
                        M33vm * (scal(i,j-3) + scal(i,j+2))

    Returns:
        scal_out: interpolated scalar field [xsize, ysize]
    """
    scal_out = jnp.zeros_like(scal).at[3:-2, 3:-2].set(
        (
            M13vm * (scal[3:-2, 2:-3] + scal[3:-2, 3:-2]) + 
            M23vm * (scal[3:-2, 1:-4] + scal[3:-2, 4:-1]) + 
            M33vm * (scal[3:-2, :-5] + scal[3:-2, 5:])
        )
    )
    
    return scal_out

@functools.partial(jax.jit, static_argnums=())
def scalar_interpolation_xy(scal: jnp.ndarray) -> jnp.ndarray:
    """
    Scalar Polynomial Interpolation for x and y directions (3rd order)

    Args:
        scal: input scalar field [xsize, ysize]

    Returns:
        tuple: (scal_x, scal_y) - both of shape [xsize, ysize]
    """
    scal_x = scalar_interpolation_x(scal)
    scal_y = scalar_interpolation_y(scal)

    return scal_x, scal_y