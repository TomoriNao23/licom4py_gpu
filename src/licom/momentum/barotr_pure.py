"""
File: barotr_pure.py
Description: Pure mathematical and physical operator implementations for the barotropic step.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-04-07

REVISION HISTORY:
    07/04/2026 - Optimized zeros_like to jnp.pad, and jnp.where to jnp.maximum for memory efficiency
"""

# Third-party imports
import jax.numpy as jnp

# Local application imports
from licom.kernel import Communication
from licom.kernel.cube import ext_scalar, ext_vector
from licom.operators import AGrid, Poly, Remap


def get_celerity(h, dzph_x, dzph_y):
    """Calculate the shallow-water gravity wave speed (celerity) on C-grid edges."""
    hx, hy = Poly.scalar_xy(h)
    celerity_x = jnp.sqrt(9.8 * (hx + dzph_x))
    celerity_y = jnp.sqrt(9.8 * (hy + dzph_y))
    return jnp.maximum(celerity_x, 10.0), jnp.maximum(celerity_y, 10.0)


def get_vel_vis_2d(celerity_x, celerity_y, h, u, v):
    """Determine 2D velocity viscosity based on celerity gradients to dampen numerical waves."""
    he, hw, hn, hs = Poly.vector(h)
    vel_vis_x_core = 4.9 / celerity_x[:, 3:-2, :] * (he[:, 2:-3, :] - hw[:, 3:-2, :])
    vel_vis_y_core = 4.9 / celerity_y[:, :, 3:-2] * (hn[:, :, 2:-3] - hs[:, :, 3:-2])
    return u.at[:, 3:-2, :].add(vel_vis_x_core), v.at[:, :, 3:-2].add(vel_vis_y_core)


def flux_calculation(h0, ub_cx, vb_cy, dzph_x, dzph_y):
    """Compute upwind mass fluxes across cell edges for the continuity equation."""
    he, hw, hn, hs = Poly.vector(h0)
    core_hx = jnp.where(ub_cx[:, 1:, :] > 0.0, he[:, :-1, :], hw[:, 1:, :])
    core_hy = jnp.where(vb_cy[:, :, 1:] > 0.0, hn[:, :, :-1], hs[:, :, 1:])

    flux_hu = dzph_x * ub_cx
    flux_hv = dzph_y * vb_cy

    flux_hu = flux_hu.at[:, 1:, :].add(core_hx * ub_cx[:, 1:, :])
    flux_hv = flux_hv.at[:, :, 1:].add(core_hy * vb_cy[:, :, 1:])

    return flux_hu, flux_hv


def fb_scheme(h0, h0_tem, beta_d):
    """Forward-Backward (FB) blending scheme for stabilizing surface height stepping."""
    return (1.0 - beta_d) * h0 + beta_d * h0_tem


def calculate_pgf(h0, rdx, rdy):
    """Calculate the Pressure Gradient Force (PGF) derived from surface elevation."""
    gradx, grady = AGrid.grad(h0, rdx, rdy)
    return -9.80 * gradx, -9.80 * grady


def calculate_pgf_vis_2d(celerity_x, celerity_y, rdx, rdy, u, v):
    """Calculate stabilized 2D horizontal viscosity directly for momentum tendencies."""
    ue, uw = Poly.vector_ew(u)
    vn, vs = Poly.vector_ns(v)

    core_x = 0.5 * celerity_x[:, 3:-2, :] * (ue[:, 2:-3, :] - uw[:, 3:-2, :])
    core_y = 0.5 * celerity_y[:, :, 3:-2] * (vn[:, :, 2:-3] - vs[:, :, 3:-2])

    diff_x = rdx[:, 3:-3, 3:-3] * (core_x[:, 1:, 3:-3] - core_x[:, :-1, 3:-3])
    diff_y = rdy[:, 3:-3, 3:-3] * (core_y[:, 3:-3, 1:] - core_y[:, 3:-3, :-1])

    pad_shape = [(0, 0)] * (u.ndim - 2) + [(3, 3), (3, 3)]
    return jnp.pad(-diff_x, pad_shape), jnp.pad(-diff_y, pad_shape)


def calculate_advection(vb_cx, ub_cy, ub_cx, vb_cy, vb_ct, ub_ct, rdx, rdy, rda, d_dx, c_dy):
    """Determine non-linear momentum advection using absolute vorticity and kinetic energy gradients."""
    vort = AGrid.vorticity(vb_cx, ub_cy, rda, d_dx, c_dy)
    kin_u = 0.5 * (ub_cx**2 + vb_cx**2)
    kin_v = 0.5 * (vb_cy**2 + ub_cy**2)
    core_advx = (
        vort[:, :-1, :] * vb_ct[:, :-1, :]
        - (kin_u[:, 1:, :] - kin_u[:, :-1, :]) * rdx[:, :-1, :]
    )
    pad_advx = [(0, 0)] * (vb_cx.ndim - 2) + [(0, 1), (0, 0)]
    advx = jnp.pad(core_advx, pad_advx)

    core_advy = (
        -vort[:, :, :-1] * ub_ct[:, :, :-1]
        - (kin_v[:, :, 1:] - kin_v[:, :, :-1]) * rdy[:, :, :-1]
    )
    pad_advy = [(0, 0)] * (vb_cy.ndim - 2) + [(0, 0), (0, 1)]
    advy = jnp.pad(core_advy, pad_advy)

    return advx, advy


def _step_rk_logic(h0, h0p, ub, vb, ubp, vbp, consts, dt, beta_d, is_laststep, px, py):
    """
    Core fractional stage execution of the Runge-Kutta operator for SPMD barotropic progression.
    Validates physical dynamics (continuity, PGF, advection) per localized tile iteration.
    """
    (dzph_x, dzph_y, rdx, rdy, a_f,
     coef, loc_i, loc_j, a_c2l, a_l2c, inner, outer,
     a_gct, a_sina, rda, c_dy, d_dx,
     pax, pxb, whx, pay, pyb, why, wgp) = consts

    # Edge Gravity Wave & Viscosity Processing
    celerity_x, celerity_y = get_celerity(h0, dzph_x, dzph_y)
    ub_ct, vb_ct, ub_cx, ub_cy, vb_cx, vb_cy = Remap.vector_trans_2d(ub, vb, a_gct, a_sina)
    ub_cx, vb_cy = get_vel_vis_2d(celerity_x, celerity_y, h0, ub_cx, vb_cy)
    flux_hu, flux_hv = flux_calculation(h0, ub_cx, vb_cy, dzph_x, dzph_y)

    if is_laststep:
        flux_hu, flux_hv = Communication.boundary_communication(flux_hu, flux_hv)

    div_out = AGrid.div(flux_hu, flux_hv, rda, c_dy, d_dx)
    new_h0 = h0p - div_out * dt
    
    # ------------------------------------------------------------------
    # ASYNC COMMUNICATION OVERLAP (Latency Hiding)
    # The XLA compiler will dispatch the network gather for new_h0 here.
    # ------------------------------------------------------------------
    new_h0 = ext_scalar(new_h0, coef, loc_i, loc_j, px, py)

    # ------------------------------------------------------------------
    # INDEPENDENT COMPUTATION (Heavy Physics)
    # Placed here so XLA can schedule them to execute asynchronously 
    # alongside the network transfer of new_h0.
    # ------------------------------------------------------------------
    advx, advy = calculate_advection(vb_cx, ub_cy, ub_cx, vb_cy, vb_ct, ub_ct, rdx, rdy, rda, d_dx, c_dy)
    vis_x, vis_y = calculate_pgf_vis_2d(celerity_x, celerity_y, rdx, rdy, ub_ct, vb_ct)

    # ------------------------------------------------------------------
    # COMMUNICATION SYNCHRONIZATION POINT 
    # Data is formally consumed causing the asynchronous barrier to yield.
    # ------------------------------------------------------------------
    h0_tem = fb_scheme(new_h0, h0, beta_d)
    pgf_u, pgf_v = calculate_pgf(h0_tem, rdx, rdy)

    # ------------------------------------------------------------------
    # MAXIMUM KERNEL FUSION (SPMD Local Execution)
    # Writing the final update as a single combined algebraic equation 
    # guarantees that XLA will compile this into precisely ONE CUDA kernel.
    # This prevents expensive VRAM register spilling and maximizes bandwidth.
    # ------------------------------------------------------------------
    new_ub = ubp + (pgf_u + advx + a_f * vb_ct + vis_x) * dt
    new_vb = vbp + (pgf_v + advy - a_f * ub_ct + vis_y) * dt

    new_ub, new_vb = ext_vector(new_ub, new_vb, coef, loc_i, loc_j, a_c2l, a_l2c, inner, outer, px, py)

    return (
        new_h0,
        new_ub,
        new_vb,
        celerity_x,
        celerity_y,
        ub_ct,
        vb_ct,
        ub_cx,
        ub_cy,
        vb_cx,
        vb_cy,
        advx,
        advy,
    )
