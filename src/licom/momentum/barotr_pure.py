"""
File: barotr_pure.py
Description: Pure mathematical and physical operator implementations for the barotropic step.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-04-07

REVISION HISTORY:
    07/04/2026 - Optimized zeros_like to jnp.pad, and jnp.where to jnp.maximum for memory efficiency
"""

import jax.numpy as jnp

from licom.mesh import Communication, Cube
from licom.operators import AGrid, Poly, Remap

def get_celerity(h, dzph_x, dzph_y):
    hx, hy = Poly.scalar_xy(h)
    celerity_x = jnp.sqrt(9.8 * (hx + dzph_x))
    celerity_y = jnp.sqrt(9.8 * (hy + dzph_y))
    return jnp.maximum(celerity_x, 10.0), \
           jnp.maximum(celerity_y, 10.0)

def get_vel_vis_2d(celerity_x, celerity_y, h, u, v):
    he, hw , hn, hs = Poly.vector(h)
    vel_vis_x_core = 4.9 / celerity_x[:, 3:-2, :] * (he[:, 2:-3, :] - hw[:, 3:-2, :])
    vel_vis_y_core = 4.9 / celerity_y[:, :, 3:-2] * (hn[:, :, 2:-3] - hs[:, :, 3:-2])
    return u.at[:, 3:-2, :].add(vel_vis_x_core), \
           v.at[:, :, 3:-2].add(vel_vis_y_core)

def flux_calculation(h0, ub_cx, vb_cy, dzph_x, dzph_y):
    he, hw, hn, hs = Poly.vector(h0)
    core_hx = jnp.where(ub_cx[:, 1:, :] > 0.0, he[:, :-1, :], hw[:, 1:, :])
    core_hy = jnp.where(vb_cy[:, :, 1:] > 0.0, hn[:, :, :-1], hs[:, :, 1:])
    
    flux_hu = dzph_x * ub_cx
    flux_hv = dzph_y * vb_cy
    
    flux_hu = flux_hu.at[:, 1:, :].add(core_hx * ub_cx[:, 1:, :])
    flux_hv = flux_hv.at[:, :, 1:].add(core_hy * vb_cy[:, :, 1:])
    
    return flux_hu, flux_hv

def fb_scheme(h0, h0_tem, beta_d):
    return (1.0 - beta_d) * h0 + beta_d * h0_tem

def calculate_pgf(h0):
    gradx, grady = AGrid.grad(h0)
    return -9.80 * gradx, -9.80 * grady

def get_pgf_vis_2d(celerity_x, celerity_y, rdx, rdy, u, v, pgf_u, pgf_v):
    ue, uw = Poly.vector_ew(u)
    vn, vs = Poly.vector_ns(v)
    
    core_x = 0.5 * celerity_x[:, 3:-2, :] * (ue[:, 2:-3, :] - uw[:, 3:-2, :])
    core_y = 0.5 * celerity_y[:, :, 3:-2] * (vn[:, :, 2:-3] - vs[:, :, 3:-2])
    
    diff_x = rdx[:, 3:-3, 3:-3] * (core_x[:, 1:, 3:-3] - core_x[:, :-1, 3:-3])
    diff_y = rdy[:, 3:-3, 3:-3] * (core_y[:, 3:-3, 1:] - core_y[:, 3:-3, :-1])
    
    return pgf_u.at[:, 3:-3, 3:-3].add(-diff_x), \
           pgf_v.at[:, 3:-3, 3:-3].add(-diff_y)

def calculate_advection(vb_cx, ub_cy, ub_cx, vb_cy, vb_ct, ub_ct, rdx, rdy):
    vort = AGrid.vorticity(vb_cx, ub_cy)
    kin_u = 0.5 * (ub_cx**2 + vb_cx**2)
    kin_v = 0.5 * (vb_cy**2 + ub_cy**2)
    core_advx = vort[:, :-1, :] * vb_ct[:, :-1, :] - (kin_u[:, 1:, :] - kin_u[:, :-1, :]) * rdx[:, :-1, :]
    pad_advx = [(0, 0)] * (vb_cx.ndim - 2) + [(0, 1), (0, 0)]
    advx = jnp.pad(core_advx, pad_advx)

    core_advy = -vort[:, :, :-1] * ub_ct[:, :, :-1] - (kin_v[:, :, 1:] - kin_v[:, :, :-1]) * rdy[:, :, :-1]
    pad_advy = [(0, 0)] * (vb_cy.ndim - 2) + [(0, 0), (0, 1)]
    advy = jnp.pad(core_advy, pad_advy)

    return advx, advy

def _step_rk_logic(h0, h0p, ub, vb, ubp, vbp, consts, dt, beta_d, is_laststep):
    dzph_x, dzph_y, pax, pxb, whx, pay, pyb, why, wgp, rdx, rdy, a_f = consts

    celerity_x, celerity_y = get_celerity(h0, dzph_x, dzph_y)
    ub_ct, vb_ct, ub_cx, ub_cy, vb_cx, vb_cy = Remap.vector_trans_2d(ub, vb)
    ub_cx, vb_cy = get_vel_vis_2d(celerity_x, celerity_y, h0, ub_cx, vb_cy)
    flux_hu, flux_hv = flux_calculation(h0, ub_cx, vb_cy, dzph_x, dzph_y)
    
    if is_laststep:
        flux_hu, flux_hv = Communication.boundary_communication(flux_hu, flux_hv)
        
    div_out = AGrid.div(flux_hu, flux_hv)
    new_h0 = h0p - div_out * dt
    new_h0 = Cube.ext_scalar(new_h0)
    
    h0_tem = fb_scheme(new_h0, h0, beta_d)
    pgf_u, pgf_v = calculate_pgf(h0_tem)
    pgf_u, pgf_v = get_pgf_vis_2d(celerity_x, celerity_y, rdx, rdy, ub_ct, vb_ct, pgf_u, pgf_v)
    
    advx, advy = calculate_advection(vb_cx, ub_cy, ub_cx, vb_cy, vb_ct, ub_ct, rdx, rdy)
    rhs_u = pgf_u + advx + a_f * vb_ct
    rhs_v = pgf_v + advy - a_f * ub_ct
    
    new_ub = ubp + rhs_u * dt
    new_vb = vbp + rhs_v * dt
    new_ub, new_vb = Cube.ext_vector(new_ub, new_vb)
    
    return new_h0, new_ub, new_vb, celerity_x, celerity_y, \
           ub_ct, vb_ct, ub_cx, ub_cy, vb_cx, vb_cy, advx, advy
