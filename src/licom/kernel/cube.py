"""
File: cube.py
Description: Pure functions for cubed-sphere grid operations, including boundary remapping
    and vector transformations. All constants and direction-dependent logic are pre-compiled
    at module/configure time. No runtime branching in hot paths.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-03-15
Updated: 2026-04-11

REVISION HISTORY:
    15/03/2026 - Initial implementation
    11/04/2026 - Pre-compiled constants/slices, eliminated try/except, removed unused params
"""

# Third-party imports
import jax
import jax.numpy as jnp

from .communication import Communication

# ==========================================
# Pre-compiled Constants (module-level, compiled once at import)
# ==========================================
_NG = 3  # Number of ghost/halo points for remapping

# Pre-compiled slice objects for edges
_SL_W = jnp.s_[..., _NG:-_NG, :_NG]
_SL_E = jnp.s_[..., _NG:-_NG, -_NG:]
_SL_S = jnp.s_[..., :_NG, _NG:-_NG]
_SL_N = jnp.s_[..., -_NG:, _NG:-_NG]


def cube_rmp(var, coef, loc_i_local, loc_j_local, px, py):
    """
    Vectorized JIT-friendly cube remapping for cubed-sphere grids.
    All runtime dynamic index offsets are eliminated.

    Args:
        var: Field array with halo points [..., ni+2*ng, nj+2*ng]
        coef, loc_i_local, loc_j_local: Pre-computed constant index/weight arrays
        px, py: Grid layout params
    """
    # Natively support arbitrary batch dimensions (e.g. nz=30, or packed stacked arrays)
    # by peeling them off from axis 1 via vmap.
    if var.ndim > 3:
        return jax.vmap(cube_rmp, in_axes=(1, None, None, None, None, None), out_axes=1)(
            var, coef, loc_i_local, loc_j_local, px, py
        )

    ix = jax.lax.axis_index("x")
    iy = jax.lax.axis_index("y")

    # ── High-Performance Pre-offset Interpolation ──────────────────────
    # Replaces dense einsum and runtime offset calc with raw take_along_axis
    # directly using pre-computed, zero-overhead host-side local offsets.

    # West edge (ix == 0)
    var_w = var[..., :, :_NG]
    loc_w = loc_i_local[_SL_W]
    new_w = coef[..., _NG:-_NG, :_NG, 0] * jnp.take_along_axis(
        var_w, loc_w, axis=-2
    ) + coef[..., _NG:-_NG, :_NG, 1] * jnp.take_along_axis(var_w, loc_w + 1, axis=-2)
    var = jax.lax.cond(ix == 0, lambda v: v.at[_SL_W].set(new_w), lambda v: v, var)

    # East edge (ix == px - 1)
    var_e = var[..., :, -_NG:]
    loc_e = loc_i_local[_SL_E]
    new_e = coef[..., _NG:-_NG, -_NG:, 0] * jnp.take_along_axis(
        var_e, loc_e, axis=-2
    ) + coef[..., _NG:-_NG, -_NG:, 1] * jnp.take_along_axis(var_e, loc_e + 1, axis=-2)
    var = jax.lax.cond(ix == px - 1, lambda v: v.at[_SL_E].set(new_e), lambda v: v, var)

    # South edge (iy == 0)
    var_s = var[..., :_NG, :]
    loc_s = loc_j_local[_SL_S]
    new_s = coef[..., :_NG, _NG:-_NG, 0] * jnp.take_along_axis(
        var_s, loc_s, axis=-1
    ) + coef[..., :_NG, _NG:-_NG, 1] * jnp.take_along_axis(var_s, loc_s + 1, axis=-1)
    var = jax.lax.cond(iy == 0, lambda v: v.at[_SL_S].set(new_s), lambda v: v, var)

    # North edge (iy == py - 1)
    var_n = var[..., -_NG:, :]
    loc_n = loc_j_local[_SL_N]
    new_n = coef[..., -_NG:, _NG:-_NG, 0] * jnp.take_along_axis(
        var_n, loc_n, axis=-1
    ) + coef[..., -_NG:, _NG:-_NG, 1] * jnp.take_along_axis(var_n, loc_n + 1, axis=-1)
    var = jax.lax.cond(iy == py - 1, lambda v: v.at[_SL_N].set(new_n), lambda v: v, var)

    # Corners — computed from edge-modified var, applied independently
    new_sw = jnp.expand_dims(var[..., _NG, :_NG], axis=-2)
    new_se = jnp.expand_dims(var[..., _NG, -_NG:], axis=-2)
    new_nw = jnp.expand_dims(var[..., -_NG - 1, :_NG], axis=-2)
    new_ne = jnp.expand_dims(var[..., -_NG - 1, -_NG:], axis=-2)

    var = jax.lax.cond(
        (iy == 0) & (ix == 0),
        lambda v: v.at[..., :_NG, :_NG].set(new_sw),
        lambda v: v,
        var,
    )
    var = jax.lax.cond(
        (iy == 0) & (ix == px - 1),
        lambda v: v.at[..., :_NG, -_NG:].set(new_se),
        lambda v: v,
        var,
    )
    var = jax.lax.cond(
        (iy == py - 1) & (ix == 0),
        lambda v: v.at[..., -_NG:, :_NG].set(new_nw),
        lambda v: v,
        var,
    )
    var = jax.lax.cond(
        (iy == py - 1) & (ix == px - 1),
        lambda v: v.at[..., -_NG:, -_NG:].set(new_ne),
        lambda v: v,
        var,
    )

    return var


class Cube:
    coef = None
    # Pre-computed local index arrays dropping the slow dynamic index calculations
    loc_i_local = None
    loc_j_local = None
    a_c2l = None
    a_l2c = None
    inner = None
    outer = None
    nx_local = 0
    ny_local = 0
    px = 1
    py = 1

    @classmethod
    def _build_local_indices(cls, loc_i, loc_j, nx_local, ny_local, nx_h, ny_h, px, py):
        """
        Pre-compute all local index coordinates (local offsets) directly on the host side.
        This decouples runtime arithmetic like `loc_i_local = loc_i - iy * ny_local` from the `cube_rmp` hot-path.
        Because the stored coordinates are precisely what is needed, retrieving interpolation points via `take_along_axis` becomes highly efficient.
        """
        # Third-party imports
        import numpy as np

        l_i = np.asarray(loc_i, dtype=np.int32)
        l_j = np.asarray(loc_j, dtype=np.int32)

        ntile = l_i.shape[0]
        i_local_global = np.zeros_like(l_i)
        j_local_global = np.zeros_like(l_j)

        for iy in range(py):
            for ix in range(px):
                # Apply block-wise offsets
                y_slc = slice(iy * ny_h, (iy + 1) * ny_h)
                x_slc = slice(ix * nx_h, (ix + 1) * nx_h)

                # West/East uses loc_i which shifts relative to the y block offset
                i_local_global[:, y_slc, x_slc] = l_i[:, y_slc, x_slc] - iy * ny_local
                # South/North uses loc_j which shifts relative to the x block offset
                j_local_global[:, y_slc, x_slc] = l_j[:, y_slc, x_slc] - ix * nx_local

        # Redistribute smoothly
        # Local application imports
        from licom.kernel.g2l import Global2Local

        spec = Global2Local.get_spec(i_local_global.shape)

        return (
            jax.device_put(
                jnp.array(i_local_global),
                jax.sharding.NamedSharding(Global2Local.mesh, spec),
            ),
            jax.device_put(
                jnp.array(j_local_global),
                jax.sharding.NamedSharding(Global2Local.mesh, spec),
            ),
        )

    @classmethod
    def configure(
        cls, coef, loc_i, loc_j, a_c2l, a_l2c, inner, outer, nx_local, ny_local, px, py
    ):
        cls.coef = coef
        cls.a_c2l = a_c2l
        cls.a_l2c = a_l2c
        cls.inner = inner
        cls.outer = outer
        cls.nx_local = nx_local
        cls.ny_local = ny_local
        cls.px = px
        cls.py = py

        nx_h = nx_local + 2 * _NG
        ny_h = ny_local + 2 * _NG
        cls.loc_i_local, cls.loc_j_local = cls._build_local_indices(
            loc_i, loc_j, nx_local, ny_local, nx_h, ny_h, px, py
        )

    @classmethod
    def ext_scalar(cls, var):
        """
        Exchange scalar values across domain boundaries.
        """
        var = Communication.update_domain(var)
        var = cube_rmp(var, cls.coef, cls.loc_i_local, cls.loc_j_local, cls.px, cls.py)
        return var

    @classmethod
    def ext_vector(cls, u, v):
        """
        Exchange vector values across domain boundaries.
        """
        ull = (cls.a_c2l[..., 0, 0] * u + cls.a_c2l[..., 0, 1] * v) * cls.inner
        vll = (cls.a_c2l[..., 1, 0] * u + cls.a_c2l[..., 1, 1] * v) * cls.inner

        # Package the two variables together into a single array for simultaneous communication.
        # We stack on axis 1 ensuring 'vtile' rigorously remains the very first axis (axis 0).
        packed = jnp.stack([ull, vll], axis=1)
        packed = cls.ext_scalar(packed)
        
        ull_new = (cls.a_l2c[..., 0, 0] * packed[:, 0] + cls.a_l2c[..., 0, 1] * packed[:, 1]) * cls.outer
        vll_new = (cls.a_l2c[..., 1, 0] * packed[:, 0] + cls.a_l2c[..., 1, 1] * packed[:, 1]) * cls.outer

        u_new = ull_new + u * cls.inner
        v_new = vll_new + v * cls.inner

        return u_new, v_new
