"""
File: barotr_jit.py
Description: JIT-compiled barotropic time stepping methods for Momentum class.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-23
"""

# Third-party imports
import jax
import jax.numpy as jnp

# Standard library imports
from typing import Tuple

# Local application imports
from operators.poly import vector_interpolation
from operators.agrid import agrid_grad, agrid_vorticity

@jax.jit
def _flux_calculation_jit(h0: jax.Array, ub_cx: jax.Array, vb_cy: jax.Array, 
                         dzph_x: jax.Array, dzph_y: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Calculate flux of <hu> & <hv> using upwind algorithm"""
    # Get interpolated h values
    he, hw, hn, hs = vector_interpolation(h0)
    
    # Upwind scheme for flux calculation
    # X-direction flux
    h_upwind_x = jnp.zeros_like(ub_cx).at[1:, :].set(
        jnp.where(
            ub_cx[1:, :] > 0.0, he[:-1, :], hw[1:, :]
        )
    )
    flux_hu = jnp.zeros_like(ub_cx).at[:, :].set(
        (h_upwind_x + dzph_x) * ub_cx
    )
    
    # Y-direction flux  
    h_upwind_y = jnp.zeros_like(vb_cy).at[:, 1:].set(
        jnp.where(vb_cy[:, 1:] > 0.0, hn[:, :-1], hs[:, 1:])
    )
    flux_hv = jnp.zeros_like(vb_cy).at[:, :].set(
        (h_upwind_y + dzph_y) * vb_cy
    )
    
    return flux_hu, flux_hv


@jax.jit
def _calculate_pgf_jit(h0: jax.Array, pax: jax.Array, pxb: jax.Array, whx: jax.Array,
                      pay: jax.Array, pyb: jax.Array, why: jax.Array, wgp: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Calculate pressure gradient force"""
    # Calculate SSH gradient
    gradx, grady = agrid_grad(h0)
    
    # PGF calculation
    grav = 9.8
    #pgf_u = (wgp - 1.0) * grav * gradx + pax + pxb - h0 * whx
    #pgf_v = (wgp - 1.0) * grav * grady + pay + pyb - h0 * why
    pgf_u = (- 1.0) * grav * gradx
    pgf_v = (- 1.0) * grav * grady
    
    return pgf_u, pgf_v


@jax.jit
def _calculate_rhs_jit(pgf_u: jax.Array, pgf_v: jax.Array, advx: jax.Array, advy: jax.Array,
                      a_f: float, vb_ct: jax.Array, ub_ct: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Calculate right-hand side of momentum equations"""
    rhs_u = pgf_u + advx + a_f * vb_ct
    rhs_v = pgf_v + advy - a_f * ub_ct
    
    return rhs_u, rhs_v


@jax.jit
def _calculate_advection_jit(vb_cx: jax.Array, ub_cy: jax.Array, ub_cx: jax.Array, 
                            vb_cy: jax.Array, vb_ct: jax.Array, ub_ct: jax.Array,
                            rdx: jax.Array, rdy: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Calculate 2D advection terms"""
    # Calculate vorticity
    vort = agrid_vorticity(vb_cx, ub_cy)
    
    # Calculate kinetic energy terms
    kinetic_u = 0.5 * (ub_cx**2 + vb_cx**2)
    kinetic_v = 0.5 * (vb_cy**2 + ub_cy**2)
    
    # Calculate advection
    advx = jnp.zeros_like(vb_cx).at[:-1, :].set(
        vort[:-1, :] * vb_ct[:-1, :] - (
            kinetic_u[1:, :] - kinetic_u[:-1, :]
        ) * rdx[:-1, :]
    )
    advy = jnp.zeros_like(vb_cy).at[:, :-1].set(
        -vort[:, :-1] * ub_ct[:, :-1] - (
            kinetic_v[:, 1:] - kinetic_v[:, :-1]
        ) * rdy[:, :-1]
    )
    
    return advx, advy


@jax.jit
def _update_ssh_jit(h0p: jax.Array, div_out: jax.Array, dt: float) -> jax.Array:
    """Update SSH field"""
    return h0p - div_out * dt


@jax.jit
def _update_velocities_jit(ubp: jax.Array, vbp: jax.Array,
                          rhs_u: jax.Array, rhs_v: jax.Array, 
                          dt: float) -> Tuple[jax.Array, jax.Array]:
    """Update velocity fields"""
    return ubp + rhs_u * dt, vbp + rhs_v * dt


@jax.jit
def _fb_scheme_jit(h0: jax.Array, h0_tem: jax.Array, beta_d: float) -> jax.Array:
    """Forward-Backward scheme"""
    return (1.0 - beta_d) * h0 + beta_d * h0_tem

@jax.jit
def _horizontal_diffusion_jit(uk: jnp.ndarray, vk: jnp.ndarray, 
                        c_dx: jnp.ndarray, d_dy: jnp.ndarray,
                        rdx: jnp.ndarray, rdy: jnp.ndarray, 
                        am_hor: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate horizontal diffusion terms for momentum equations.
    
    Original Fortran code:
    do j = jsd+1, jed-1
      do i = isd+1, ied-1
         hduk(i,j) = ((uk(i+1,j) - uk(i,j))/dg%c_dx(i+1,j) -   &
                      ((uk(i,j) - uk(i-1,j))/dg%c_dx(i,j)))*dg%rdx(i,j) + &
                     ((uk(i,j+1) - uk(i,j))/dg%d_dy(i,j+1) -   &
                      ((uk(i,j) - uk(i,j-1))/dg%d_dy(i,j)))*dg%rdy(i,j) 
         hdvk(i,j) = ((vk(i+1,j) - vk(i,j))/dg%c_dx(i+1,j) -   &
                      (vk(i,j) - vk(i-1,j))/dg%c_dx(i,j))*dg%rdx(i,j) + &
                     ((vk(i,j+1) - vk(i,j))/dg%d_dy(i,j+1) -   &
                      (vk(i,j) - vk(i,j-1))/dg%d_dy(i,j))*dg%rdy(i,j) 
         hduk(i,j) = hduk(i,j)*fg%am_hor
         hdvk(i,j) = hdvk(i,j)*fg%am_hor
      end do
    end do
    
    Args:
        uk: U velocity field
        vk: V velocity field  
        c_dx: Grid spacing in x on C-grid
        d_dy: Grid spacing in y on D-grid
        rdx: Reciprocal dx
        rdy: Reciprocal dy
        am_hor: Horizontal mixing coefficient
        
    Returns:
        tuple: (hduk, hdvk) - Horizontal diffusion terms
    """
    
    # Initialize output arrays
    hduk = jnp.zeros_like(uk)
    hdvk = jnp.zeros_like(vk)
    
    # U-component diffusion (interior points only: [1:-1, 1:-1])
    # X-direction term
    u_diffx = ((uk[2:, 1:-1] - uk[1:-1, 1:-1]) / c_dx[2:, 1:-1] - 
               (uk[1:-1, 1:-1] - uk[:-2, 1:-1]) / c_dx[1:-1, 1:-1]) * rdx[1:-1, 1:-1]
    
    # Y-direction term  
    u_diffy = ((uk[1:-1, 2:] - uk[1:-1, 1:-1]) / d_dy[1:-1, 2:] - 
               (uk[1:-1, 1:-1] - uk[1:-1, :-2]) / d_dy[1:-1, 1:-1]) * rdy[1:-1, 1:-1]
    
    hduk = hduk.at[1:-1, 1:-1].set((u_diffx + u_diffy) * am_hor)
    
    # V-component diffusion (interior points only: [1:-1, 1:-1])
    # X-direction term
    v_diffx = ((vk[2:, 1:-1] - vk[1:-1, 1:-1]) / c_dx[2:, 1:-1] - 
               (vk[1:-1, 1:-1] - vk[:-2, 1:-1]) / c_dx[1:-1, 1:-1]) * rdx[1:-1, 1:-1]
    
    # Y-direction term
    v_diffy = ((vk[1:-1, 2:] - vk[1:-1, 1:-1]) / d_dy[1:-1, 2:] - 
               (vk[1:-1, 1:-1] - vk[1:-1, :-2]) / d_dy[1:-1, 1:-1]) * rdy[1:-1, 1:-1]
    
    hdvk = hdvk.at[1:-1, 1:-1].set((v_diffx + v_diffy) * am_hor)
    
    return hduk, hdvk