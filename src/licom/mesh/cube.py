"""
File: cube.py
Description: Pure functions for cubed-sphere grid operations, including boundary remapping
    and vector transformations. All functions are pure and operate directly on sharded
    arrays (e.g., shape (ntile, ni, nj)) without internal JIT or vmap.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-03-15
Updated: 2026-03-15
"""

import jax
import jax.numpy as jnp
from .communication import Communication


def cube_rmp(var, coef, loc_i, loc_j, inner, nx_local, ny_local, px, py):
    """
    Vectorized JIT-friendly cube remapping for cubed-sphere grids.
    Operates on full sharded arrays (ntile, ni, nj) or robust ... slicing.

    Assumptions:
    - `var` already includes halo points, and boundaries correspond to local tile boundaries:
      south: [..., :ng], north: [..., -ng:], west: [..., :ng, :], east: [..., -ng:, :]
    - `loc_i`, `loc_j` are the pre-computed arrays of local indices along the remapping direction.
    - `coef[..., 0:2]` stores weights for the two-point stencil.
    """
    nord = 2
    ng = 3
    
    try:
        ix = jax.lax.axis_index('x')
        iy = jax.lax.axis_index('y')
    except NameError:
        ix, iy = 0, 0

    loc_i = loc_i.astype(jnp.int32) - iy * ny_local
    loc_j = loc_j.astype(jnp.int32) - ix * nx_local
    k = jnp.arange(nord, dtype=jnp.int32)

    # West edge (x=0 limit -> slice :ng on axis -1)
    sl_w = jnp.s_[..., ng:-ng, :ng]
    idx0_w = loc_i[sl_w][..., None] + k
    vals_w = jnp.take_along_axis(var[..., :, :ng][..., None], idx0_w, axis=-3)
    new_w = jnp.sum(vals_w * coef[..., ng:-ng, :ng, :], axis=-1)
    var = jax.lax.cond(
        ix == 0,
        lambda v: v.at[sl_w].set(new_w),
        lambda v: v,
        var
    )

    # East edge (x=max limit -> slice -ng: on axis -1)
    sl_e = jnp.s_[..., ng:-ng, -ng:]
    idx0_e = loc_i[sl_e][..., None] + k
    vals_e = jnp.take_along_axis(var[..., :, -ng:][..., None], idx0_e, axis=-3)
    new_e = jnp.sum(vals_e * coef[..., ng:-ng, -ng:, :], axis=-1)
    var = jax.lax.cond(
        ix == px - 1,
        lambda v: v.at[sl_e].set(new_e),
        lambda v: v,
        var
    )

    # South edge (y=0 limit -> slice :ng on axis -2)
    sl_s = jnp.s_[..., :ng, ng:-ng]
    idx1_s = loc_j[sl_s][..., None] + k
    vals_s = jnp.take_along_axis(var[..., :ng, :][..., None], idx1_s, axis=-2)
    new_s = jnp.sum(vals_s * coef[..., :ng, ng:-ng, :], axis=-1)
    var = jax.lax.cond(
        iy == 0,
        lambda v: v.at[sl_s].set(new_s),
        lambda v: v,
        var
    )

    # North edge (y=max limit -> slice -ng: on axis -2)
    sl_n = jnp.s_[..., -ng:, ng:-ng]
    idx1_n = loc_j[sl_n][..., None] + k
    vals_n = jnp.take_along_axis(var[..., -ng:, :][..., None], idx1_n, axis=-2)
    new_n = jnp.sum(vals_n * coef[..., -ng:, ng:-ng, :], axis=-1)
    var = jax.lax.cond(
        iy == py - 1,
        lambda v: v.at[sl_n].set(new_n),
        lambda v: v,
        var
    )

    # Corners
    # SW (South-West: iy=0, ix=0 -> y[:ng], x[:ng])
    new_sw = jnp.expand_dims(var[..., ng, :ng], axis=-2)
    var = jax.lax.cond(
        (iy == 0) & (ix == 0),
        lambda v: v.at[..., :ng, :ng].set(new_sw),
        lambda v: v,
        var
    )
    # SE (South-East: iy=0, ix=px-1 -> y[:ng], x[-ng:])
    new_se = jnp.expand_dims(var[..., ng, -ng:], axis=-2)
    var = jax.lax.cond(
        (iy == 0) & (ix == px - 1),
        lambda v: v.at[..., :ng, -ng:].set(new_se),
        lambda v: v,
        var
    )
    # NW (North-West: iy=py-1, ix=0 -> y[-ng:], x[:ng])
    new_nw = jnp.expand_dims(var[..., -ng-1, :ng], axis=-2)
    var = jax.lax.cond(
        (iy == py - 1) & (ix == 0),
        lambda v: v.at[..., -ng:, :ng].set(new_nw),
        lambda v: v,
        var
    )
    # NE (North-East: iy=py-1, ix=px-1 -> y[-ng:], x[-ng:])
    new_ne = jnp.expand_dims(var[..., -ng-1, -ng:], axis=-2)
    var = jax.lax.cond(
        (iy == py - 1) & (ix == px - 1),
        lambda v: v.at[..., -ng:, -ng:].set(new_ne),
        lambda v: v,
        var
    )

    return var

class Cube:
    coef = None
    loc_i = None
    loc_j = None
    a_c2l = None
    a_l2c = None
    inner = None
    outer = None
    nx_local = 0
    ny_local = 0
    px = 1
    py = 1

    @classmethod
    def configure(cls, coef, loc_i, loc_j, a_c2l, a_l2c, inner, outer, nx_local, ny_local, px, py):
        cls.coef = coef
        cls.loc_i = loc_i
        cls.loc_j = loc_j
        cls.a_c2l = a_c2l
        cls.a_l2c = a_l2c
        cls.inner = inner
        cls.outer = outer
        cls.nx_local = nx_local
        cls.ny_local = ny_local
        cls.px = px
        cls.py = py

    @classmethod
    def ext_scalar(cls, var):
        """
        Exchange scalar values across domain boundaries.
        """
        var = Communication.update_domain(var)
        var = cube_rmp(var, cls.coef, cls.loc_i, cls.loc_j, cls.inner, cls.nx_local, cls.ny_local, cls.px, cls.py)
        return var

    @classmethod
    def ext_vector(cls, u, v):
        """
        Exchange vector values across domain boundaries.
        """
        ull = (cls.a_c2l[..., 0, 0] * u + cls.a_c2l[..., 0, 1] * v) * cls.inner
        vll = (cls.a_c2l[..., 1, 0] * u + cls.a_c2l[..., 1, 1] * v) * cls.inner

        ull = cls.ext_scalar(ull)
        vll = cls.ext_scalar(vll)

        ull_new = (cls.a_l2c[..., 0, 0] * ull + cls.a_l2c[..., 0, 1] * vll) * cls.outer
        vll_new = (cls.a_l2c[..., 1, 0] * ull + cls.a_l2c[..., 1, 1] * vll) * cls.outer

        u_new = ull_new + u * cls.inner
        v_new = vll_new + v * cls.inner

        return u_new, v_new