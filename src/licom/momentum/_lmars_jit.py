"""
File: lmars.py
Description: LMARS methods for Momentum class.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-23
Updated: 2025-09-24
REVISION HISTORY:
    2025-09-23 - Chtholly added LMARS methods
    2025-09-24 - Chtholly added JIT-compiled methods
"""

# Third-party imports
import jax
import jax.numpy as jnp

# Standard library imports
from typing import Tuple

# Local application imports
from backend.calculation.field import Field
from momentum.barotr import jax
from operators.poly import scalar_interpolation_xy, vector_interpolation
from operators.poly import vector_interpolation_ew, vector_interpolation_ns

@jax.jit
def get_celerity_jit(
    h : Field.datatype, 
    dzph_x : Field.datatype, dzph_y : Field.datatype) -> Tuple[Field.datatype, Field.datatype]:
    """
    Get the celerity of the ocean current.
    TODO: need to add the topography.

    Formula:
        a_x = sqrt(g * h_x + dzph_x)
        a_y = sqrt(g * h_y + dzph_y)

    a = 10.0 m/swhere a < 10.0 m/s
    """

    # Initialize
    celerity_x = Field.new('2d')
    celerity_y = Field.new('2d')

    # get the interpolated ocean depth
    hx, hy = scalar_interpolation_xy(h)

    # get the celerity of the ocean current
    celerity_x = celerity_x.at[:,:].set(
        jnp.sqrt(9.8 * (hx + dzph_x))
    )
    celerity_y = celerity_y.at[:,:].set(
        jnp.sqrt(9.8 * (hy + dzph_y))
    )

    # limit the celerity limit to 10.0 m/s
    celerity_x = celerity_x.at[:,:].set(
        jnp.where(
            celerity_x[:,:]<10.0,
            10.0,
            celerity_x[:,:]
        )
    )
    celerity_y = celerity_y.at[:,:].set(
        jnp.where(
            celerity_y[:,:]<10.0,
            10.0,
            celerity_y[:,:]
        )
    )
    

    return celerity_x, celerity_y

@jax.jit
def get_vel_vis_2d_jit(
    celerity_x : Field.datatype, celerity_y : Field.datatype, 
    h : Field.datatype, 
    u : Field.datatype, v : Field.datatype) -> Tuple[Field.datatype, Field.datatype]:
    """
    Get the Viscosity.velocity of the ocean current.
    TODO: need to add the topography.

    Formula:
        vel_vis_x = g * (h_e(i-1) - h_w(i)) * 0.5 / a
        vel_vis_y = g * (h_n(j-1) - h_s(j)) * 0.5 / a
    """

    # Initialize
    vel_vis_x = Field.new('2d')
    vel_vis_y = Field.new('2d')

    # get the interpolated ocean depth
    he, hw , hn, hs = vector_interpolation(h)

    # get the Viscosity.velocity of the ocean current
    vel_vis_x = vel_vis_x.at[3:-2,:].set(
        4.9 / celerity_x[3:-2,:] * (
            he[2:-3,:] - hw[3:-2,:]
        )
    )
    vel_vis_y = vel_vis_y.at[:,3:-2].set(
        4.9 / celerity_y[:,3:-2] * (
            hn[:,2:-3] - hs[:,3:-2]
        )
    )
    u = u.at[3:-2,:].set(u[3:-2,:] + vel_vis_x[3:-2,:])
    v = v.at[:,3:-2].set(v[:,3:-2] + vel_vis_y[:,3:-2])

    return u, v


@jax.jit
def get_pgf_vis_2d_jit(
    celerity_x : Field.datatype, celerity_y : Field.datatype, 
    rdx : Field.datatype, rdy : Field.datatype,
    u : Field.datatype, v : Field.datatype, 
    pgf_u : Field.datatype, pgf_v : Field.datatype) -> Tuple[Field.datatype, Field.datatype]:
    """
    Get the Viscosity.pressure gradient force of the ocean current.

    Formula:
        vel_vis_x = 0.5 * a * (u_e(i-1) - u_w(i))
        vel_vis_y = 0.5 * a * (v_n(j-1) - v_s(j))
    """

    # Initialize
    vel_vis_x = Field.new('2d')
    vel_vis_y = Field.new('2d')

    # get the interpolated ocean current
    ue, uw = vector_interpolation_ew(u)
    vn, vs = vector_interpolation_ns(v)
    
    # get the Viscosity.pressure of the ocean current
    vel_vis_x = vel_vis_x.at[3:-2,:].set(
        0.5 * celerity_x[3:-2,:] * (
            ue[2:-3,:] - uw[3:-2,:]
        )
    )
    vel_vis_y = vel_vis_y.at[:,3:-2].set(
        0.5 * celerity_y[:,3:-2] * (
            vn[:,2:-3] - vs[:,3:-2]
        )
    )

    # get the Viscosity.pressure gradient force of the ocean current
    vel_vis_x = vel_vis_x.at[3:-3,3:-3].set(
        rdx[3:-3,3:-3] * (
            vel_vis_x[4:-2,3:-3] - vel_vis_x[3:-3,3:-3]
        )
    )
    vel_vis_y = vel_vis_y.at[3:-3,3:-3].set(
        rdy[3:-3,3:-3] * (
            vel_vis_y[3:-3,4:-2] - vel_vis_y[3:-3,3:-3]
        )
    )

    # Add the Viscosity.pressure gradient force of the ocean current
    pgf_u = pgf_u.at[3:-3,3:-3].set(pgf_u[3:-3,3:-3] - vel_vis_x[3:-3,3:-3])
    pgf_v = pgf_v.at[3:-3,3:-3].set(pgf_v[3:-3,3:-3] - vel_vis_y[3:-3,3:-3])

    return pgf_u, pgf_v