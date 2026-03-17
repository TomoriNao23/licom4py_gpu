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


def cube_rmp(var, coef, loc_i, loc_j, inner):
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
    ng = Communication.halo if hasattr(Communication, "halo") and Communication.halo > 0 else 3

    loc_i = loc_i.astype(jnp.int32)
    loc_j = loc_j.astype(jnp.int32)
    k = jnp.arange(nord, dtype=jnp.int32)

    # South edge
    sl_s = jnp.s_[..., ng:-ng, :ng]
    idx0_s = loc_i[sl_s][..., None] + k
    vals_s = jnp.take_along_axis(var[..., :, :ng][..., None], idx0_s, axis=-3)
    var = var.at[sl_s].set(jnp.sum(vals_s * coef[..., ng:-ng, :ng, :], axis=-1))

    # North edge
    sl_n = jnp.s_[..., ng:-ng, -ng:]
    idx0_n = loc_i[sl_n][..., None] + k
    vals_n = jnp.take_along_axis(var[..., :, -ng:][..., None], idx0_n, axis=-3)
    var = var.at[sl_n].set(jnp.sum(vals_n * coef[..., ng:-ng, -ng:, :], axis=-1))

    # West edge
    sl_w = jnp.s_[..., :ng, ng:-ng]
    idx1_w = loc_j[sl_w][..., None] + k
    vals_w = jnp.take_along_axis(var[..., :ng, :][..., None], idx1_w, axis=-2)
    var = var.at[sl_w].set(jnp.sum(vals_w * coef[..., :ng, ng:-ng, :], axis=-1))

    # East edge
    sl_e = jnp.s_[..., -ng:, ng:-ng]
    idx1_e = loc_j[sl_e][..., None] + k
    vals_e = jnp.take_along_axis(var[..., -ng:, :][..., None], idx1_e, axis=-2)
    var = var.at[sl_e].set(jnp.sum(vals_e * coef[..., -ng:, ng:-ng, :], axis=-1))

    # Corners
    # SW
    var = var.at[..., :ng, :ng].set(jnp.expand_dims(var[..., ng, :ng], axis=-2))
    # SE
    var = var.at[..., -ng:, :ng].set(jnp.expand_dims(var[..., -ng-1, :ng], axis=-2))
    # NE
    var = var.at[..., -ng:, -ng:].set(jnp.expand_dims(var[..., -ng-1, -ng:], axis=-2))
    # NW
    var = var.at[..., :ng, -ng:].set(jnp.expand_dims(var[..., ng, -ng:], axis=-2))

    return var

class Cube:
    coef = None
    loc_i = None
    loc_j = None
    a_c2l = None
    a_l2c = None
    inner = None
    outer = None

    @classmethod
    def configure(cls, coef, loc_i, loc_j, a_c2l, a_l2c, inner, outer):
        cls.coef = coef
        cls.loc_i = loc_i
        cls.loc_j = loc_j
        cls.a_c2l = a_c2l
        cls.a_l2c = a_l2c
        cls.inner = inner
        cls.outer = outer

    @classmethod
    def ext_scalar(cls, var):
        """
        Exchange scalar values across domain boundaries.
        """
        var = Communication.update_domain(var)
        var = cube_rmp(var, cls.coef, cls.loc_i, cls.loc_j, cls.inner)
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


