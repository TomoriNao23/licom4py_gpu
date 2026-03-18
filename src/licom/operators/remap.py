"""
File: remap.py
Description: Grid remapping functions as static methods of the Remap class.
    Handles A-grid, C-grid, D-grid remapping and 2D vector transformation.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-20
Updated: 2026-03-19

REVISION HISTORY:
    20/09/2025 - Initial implementation
    21/09/2025 - Added upwind scheme
    19/03/2026 - Refactor imports to package-level paths
    19/03/2026 - Wrap functions as Remap static methods
"""

# Third-party imports
import functools
import jax
import jax.numpy as jnp

# Standard library imports
from typing import Tuple

# Local application imports
from .poly import Poly
from licom.duogrid import Dg
from licom.mesh import Communication


class Remap:
    """Grid remapping operators: A-grid, C-grid, D-grid, and vector transformation."""

    @staticmethod
    def to_a_grid(u: jnp.ndarray, v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Remap velocity to A-grid (contravariant transformation).

        Args:
            u, v: velocity components [..., xsize, ysize]

        Returns:
            (uct, vct) - contravariant velocities on A-grid
        """
        uct = jnp.zeros_like(u).at[..., :].set(
            (Dg.a_gct[..., 0, 0] * u + Dg.a_gct[..., 0, 1] * v) * Dg.a_sina
        )
        vct = jnp.zeros_like(v).at[..., :].set(
            (Dg.a_gct[..., 1, 0] * u + Dg.a_gct[..., 1, 1] * v) * Dg.a_sina
        )
        return uct, vct

    @staticmethod
    def to_c_grid(u: jnp.ndarray, v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Remap velocity to C-grid.

        Args:
            u, v: velocity components [..., xsize, ysize]

        Returns:
            (uc, vc) - C-grid remapped velocities
        """
        ue, uw = Poly.vector_ew(u)
        un, us = Poly.vector_ns(v)

        uc = jnp.zeros_like(u).at[..., 3:-2, :].set(
            0.5 * (ue[..., 2:-3, :] + uw[..., 3:-2, :])
        )
        vc = jnp.zeros_like(v).at[..., :, 3:-2].set(
            0.5 * (un[..., :, 2:-3] + us[..., :, 3:-2])
        )
        return uc, vc

    @staticmethod
    def to_d_grid(u: jnp.ndarray, v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Remap velocity to D-grid.

        Args:
            u, v: velocity components [..., xsize, ysize]

        Returns:
            (ud, vd) - D-grid remapped velocities
        """
        un, us = Poly.vector_ns(u)
        ve, vw = Poly.vector_ew(v)

        ud = jnp.zeros_like(u).at[..., 3:-2, :].set(
            0.5 * (un[..., 2:-3, :] + us[..., 3:-2, :])
        )
        vd = jnp.zeros_like(v).at[..., :, 3:-2].set(
            0.5 * (ve[..., :, 2:-3] + vw[..., :, 3:-2])
        )
        return ud, vd

    @staticmethod
    @functools.partial(jax.jit, static_argnums=())
    def to_d_grid_upwind(
        u: jnp.ndarray, v: jnp.ndarray,
        uc: jnp.ndarray, vc: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Remap velocity to D-grid using upwind scheme.

        Args:
            u, v: velocity components [..., xsize, ysize]
            uc, vc: contravariant velocities for upwind selection

        Returns:
            (ud, vd) - D-grid velocities with upwind scheme
        """
        un, us = Poly.vector_ns(u)
        ve, vw = Poly.vector_ew(v)

        vd = jnp.zeros_like(v).at[..., 3:-2, :].set(
            jnp.where(
                uc[..., 3:-2, :] > 0.0,
                ve[..., 2:-3, :],
                vw[..., 3:-2, :],
            )
        )
        ud = jnp.zeros_like(u).at[..., :, 3:-2].set(
            jnp.where(
                vc[..., :, 3:-2] > 0.0,
                un[..., :, 2:-3],
                us[..., :, 3:-2],
            )
        )
        return ud, vd

    @staticmethod
    def vector_trans_2d(
        u: jnp.ndarray, v: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Full 2D vector transformation: A-grid → C-grid (with communication) → D-grid (upwind).

        Args:
            u, v: velocity components [..., xsize, ysize]

        Returns:
            (uct, vct, ub_cx, ub_cy, vb_cx, vb_cy)
        """
        uct, vct   = Remap.to_a_grid(u, v)
        ub_cx, vb_cy = Remap.to_c_grid(uct, vct)
        ub_cx, vb_cy = Communication.boundary_communication(ub_cx, vb_cy)
        ub_cy, vb_cx = Remap.to_d_grid_upwind(u, v, ub_cx, vb_cy)
        return uct, vct, ub_cx, ub_cy, vb_cx, vb_cy