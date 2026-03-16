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
from duogrid import Dg

def _transform(u, v, k, range_mask):
    """
    Transform the vector values between local and canonical coordinates.
    """
    ull = (k[..., 0, 0] * u + k[..., 0, 1] * v) * range_mask
    vll = (k[..., 1, 0] * u + k[..., 1, 1] * v) * range_mask
    return ull, vll

def cube_rmp(var, coef, loc_arr):
    """
    Apply boundary remapping across all tiles using precomputed coefficients.
    Convert halo/edge values on a cubed-sphere tile using precomputed
    k2e mappings and coefficients.

    Args:
        var: (ntile, ni, nj) Array for which boundary remapping is applied.
        coef: (ntile, ni, nj, nord) k2e coefficients array.
        loc_arr: (ntile, ni, nj) k2e location array.

    Returns:
        The updated `var` after remapping.
    """
    # Hardcoded indices for C96 grid as requested
    isd, jsd, ied, jed = -2, -2, 99, 99
    is_, js, ie, je = 1, 1, 96, 96
    
    # Boundary flags for each tile
    rmp_s, rmp_n, rmp_w, rmp_e = True, True, True, True
    
    var_kik = var
    offset = 1
    ntile = var.shape[0]
    ti = jnp.arange(ntile)[:, jnp.newaxis] # (ntile, 1) to broadcast with spatial indices

    # South boundary
    if rmp_s:
        # ii = 1
        j = js - 1
        ii = jnp.arange(is_, ie + 1)
        locs = loc_arr[:, ii - isd, j - jsd].astype(int)
        los = locs - offset
        var = var.at[:, ii - isd, j - jsd].set(0.0)
        var = var.at[:, ii - isd, j - jsd].add(var_kik[ti, los + 1 - isd, j - jsd] * coef[:, ii - isd, j - jsd, 0])
        var = var.at[:, ii - isd, j - jsd].add(var_kik[ti, los + 2 - isd, j - jsd] * coef[:, ii - isd, j - jsd, 1])
        # ii = 2
        j = js - 2
        locs = loc_arr[:, ii - isd, j - jsd].astype(int)
        los = locs - offset
        var = var.at[:, ii - isd, j - jsd].set(0.0)
        var = var.at[:, ii - isd, j - jsd].add(var_kik[ti, los + 1 - isd, j - jsd] * coef[:, ii - isd, j - jsd, 0])
        var = var.at[:, ii - isd, j - jsd].add(var_kik[ti, los + 2 - isd, j - jsd] * coef[:, ii - isd, j - jsd, 1])
        # ii = 3
        j = js - 3
        locs = loc_arr[:, ii - isd, j - jsd].astype(int)
        los = locs - offset
        var = var.at[:, ii - isd, j - jsd].set(0.0)
        var = var.at[:, ii - isd, j - jsd].add(var_kik[ti, los + 1 - isd, j - jsd] * coef[:, ii - isd, j - jsd, 0])
        var = var.at[:, ii - isd, j - jsd].add(var_kik[ti, los + 2 - isd, j - jsd] * coef[:, ii - isd, j - jsd, 1])

    # North boundary
    if rmp_n:
        ii = jnp.arange(is_, ie + 1)
        # ii = 1
        j = je + 1
        locs = loc_arr[:, ii - isd, j - jsd].astype(int)
        los = locs - offset
        var = var.at[:, ii - isd, j - jsd].set(0.0)
        var = var.at[:, ii - isd, j - jsd].add(var_kik[ti, los + 1 - isd, j - jsd] * coef[:, ii - isd, j - jsd, 0])
        var = var.at[:, ii - isd, j - jsd].add(var_kik[ti, los + 2 - isd, j - jsd] * coef[:, ii - isd, j - jsd, 1])
        # ii = 2
        j = je + 2
        locs = loc_arr[:, ii - isd, j - jsd].astype(int)
        los = locs - offset
        var = var.at[:, ii - isd, j - jsd].set(0.0)
        var = var.at[:, ii - isd, j - jsd].add(var_kik[ti, los + 1 - isd, j - jsd] * coef[:, ii - isd, j - jsd, 0])
        var = var.at[:, ii - isd, j - jsd].add(var_kik[ti, los + 2 - isd, j - jsd] * coef[:, ii - isd, j - jsd, 1])
        # ii = 3
        j = je + 3
        locs = loc_arr[:, ii - isd, j - jsd].astype(int)
        los = locs - offset
        var = var.at[:, ii - isd, j - jsd].set(0.0)
        var = var.at[:, ii - isd, j - jsd].add(var_kik[ti, los + 1 - isd, j - jsd] * coef[:, ii - isd, j - jsd, 0])
        var = var.at[:, ii - isd, j - jsd].add(var_kik[ti, los + 2 - isd, j - jsd] * coef[:, ii - isd, j - jsd, 1])
              
    # West boundary
    if rmp_w:
        jj = jnp.arange(js, je + 1)
        # ii = 1
        i = is_ - 1
        locs = loc_arr[:, i - isd, jj - jsd].astype(int)
        los = locs - offset
        var = var.at[:, i - isd, jj - jsd].set(0.0)
        var = var.at[:, i - isd, jj - jsd].add(var_kik[ti, i - isd, los + 1 - jsd] * coef[:, i - isd, jj - jsd, 0])
        var = var.at[:, i - isd, jj - jsd].add(var_kik[ti, i - isd, los + 2 - jsd] * coef[:, i - isd, jj - jsd, 1])
        # ii = 2
        i = is_ - 2
        locs = loc_arr[:, i - isd, jj - jsd].astype(int)
        los = locs - offset
        var = var.at[:, i - isd, jj - jsd].set(0.0)
        var = var.at[:, i - isd, jj - jsd].add(var_kik[ti, i - isd, los + 1 - jsd] * coef[:, i - isd, jj - jsd, 0])
        var = var.at[:, i - isd, jj - jsd].add(var_kik[ti, i - isd, los + 2 - jsd] * coef[:, i - isd, jj - jsd, 1])
        # ii = 3
        i = is_ - 3
        locs = loc_arr[:, i - isd, jj - jsd].astype(int)
        los = locs - offset
        var = var.at[:, i - isd, jj - jsd].set(0.0)
        var = var.at[:, i - isd, jj - jsd].add(var_kik[ti, i - isd, los + 1 - jsd] * coef[:, i - isd, jj - jsd, 0])
        var = var.at[:, i - isd, jj - jsd].add(var_kik[ti, i - isd, los + 2 - jsd] * coef[:, i - isd, jj - jsd, 1])

    # East boundary
    if rmp_e:
        jj = jnp.arange(js, je + 1)
        # ii = 1
        i = ie + 1
        locs = loc_arr[:, i - isd, jj - jsd].astype(int)
        los = locs - offset
        var = var.at[:, i - isd, jj - jsd].set(0.0)
        var = var.at[:, i - isd, jj - jsd].add(var_kik[ti, i - isd, los + 1 - jsd] * coef[:, i - isd, jj - jsd, 0])
        var = var.at[:, i - isd, jj - jsd].add(var_kik[ti, i - isd, los + 2 - jsd] * coef[:, i - isd, jj - jsd, 1])
        # ii = 2
        i = ie + 2
        locs = loc_arr[:, i - isd, jj - jsd].astype(int)
        los = locs - offset
        var = var.at[:, i - isd, jj - jsd].set(0.0)
        var = var.at[:, i - isd, jj - jsd].add(var_kik[ti, i - isd, los + 1 - jsd] * coef[:, i - isd, jj - jsd, 0])
        var = var.at[:, i - isd, jj - jsd].add(var_kik[ti, i - isd, los + 2 - jsd] * coef[:, i - isd, jj - jsd, 1])
        # ii = 3
        i = ie + 3
        locs = loc_arr[:, i - isd, jj - jsd].astype(int)
        los = locs - offset
        var = var.at[:, i - isd, jj - jsd].set(0.0)
        var = var.at[:, i - isd, jj - jsd].add(var_kik[ti, i - isd, los + 1 - jsd] * coef[:, i - isd, jj - jsd, 0])
        var = var.at[:, i - isd, jj - jsd].add(var_kik[ti, i - isd, los + 2 - jsd] * coef[:, i - isd, jj - jsd, 1])

    # Copy corners
    # sw
    var = var.at[:, 0 : is_ - isd, 0 : js - jsd].set(var[:, is_ - isd : is_ - isd + 1, 0 : js - jsd])
    # se
    var = var.at[:, ie + 1 - isd : ied + 1 - isd, 0 : js - jsd].set(var[:, ie - isd : ie - isd + 1, 0 : js - jsd])
    # ne
    var = var.at[:, ie + 1 - isd : ied + 1 - isd, je + 1 - jsd : jed + 1 - jsd].set(var[:, ie - isd : ie - isd + 1, je + 1 - jsd : jed + 1 - jsd])
    # nw
    var = var.at[:, 0 : is_ - isd, je + 1 - jsd : jed + 1 - jsd].set(var[:, is_ - isd : is_ - isd + 1, je + 1 - jsd : jed + 1 - jsd])

    return var

def ext_scalar(var):
    """
    Extend a scalar field across domain boundaries.
    """
    var = Communication.update_domain(var)
    return cube_rmp(var, Dg.k2e_coef, Dg.k2e_loc)

def ext_vector(u, v):
    """
    Extend a vector field across domain boundaries.
    """
    ull, vll = _transform(u, v, Dg.a_c2l, Dg.inner)
    ull = ext_scalar(ull)
    vll = ext_scalar(vll)
    ull, vll = _transform(ull, vll, Dg.a_l2c, Dg.outer)
    
    ull = ull + u * Dg.inner
    vll = vll + v * Dg.inner
    return ull, vll
