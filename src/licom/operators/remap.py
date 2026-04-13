"""
File: remap.py
Description: Grid remapping functions as pure static methods.
    All grid data passed explicitly — no singleton access.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-20
Updated: 2026-04-13

REVISION HISTORY:
    20/09/2025 - Initial implementation
    21/09/2025 - Added upwind scheme
    19/03/2026 - Refactor imports to package-level paths
    19/03/2026 - Wrap functions as Remap static methods
    07/04/2026 - Replaced zeros_like buffer scatters with native jnp.pad
    13/04/2026 - Converted to explicit parameter passing
"""

# Standard library imports
from typing import Tuple

# Third-party imports
import jax.numpy as jnp

# Local application imports
from licom.kernel import Communication

from .poly import Poly


class Remap:
    """Grid remapping operators: A-grid, C-grid, D-grid, and vector transformation."""

    @staticmethod
    def to_a_grid(u, v, a_gct, a_sina):
        """Remap velocity to A-grid (contravariant transformation)."""
        uct = (a_gct[..., 0, 0] * u + a_gct[..., 0, 1] * v) * a_sina
        vct = (a_gct[..., 1, 0] * u + a_gct[..., 1, 1] * v) * a_sina
        return uct, vct

    @staticmethod
    def to_c_grid(u, v):
        """Remap velocity to C-grid."""
        ue, uw = Poly.vector_ew(u)
        un, us = Poly.vector_ns(v)

        core_uc = 0.5 * (ue[..., 2:-3, :] + uw[..., 3:-2, :])
        pad_uc = [(0, 0)] * (u.ndim - 2) + [(3, 2), (0, 0)]
        uc = jnp.pad(core_uc, pad_uc)

        core_vc = 0.5 * (un[..., :, 2:-3] + us[..., :, 3:-2])
        pad_vc = [(0, 0)] * (v.ndim - 2) + [(0, 0), (3, 2)]
        vc = jnp.pad(core_vc, pad_vc)

        return uc, vc

    @staticmethod
    def to_d_grid(u, v):
        """Remap velocity to D-grid."""
        un, us = Poly.vector_ns(u)
        ve, vw = Poly.vector_ew(v)

        core_ud = 0.5 * (un[..., 2:-3, :] + us[..., 3:-2, :])
        pad_ud = [(0, 0)] * (u.ndim - 2) + [(3, 2), (0, 0)]
        ud = jnp.pad(core_ud, pad_ud)

        core_vd = 0.5 * (ve[..., :, 2:-3] + vw[..., :, 3:-2])
        pad_vd = [(0, 0)] * (v.ndim - 2) + [(0, 0), (3, 2)]
        vd = jnp.pad(core_vd, pad_vd)

        return ud, vd

    @staticmethod
    def to_d_grid_upwind(u, v, uc, vc):
        """Remap velocity to D-grid using upwind scheme."""
        un, us = Poly.vector_ns(u)
        ve, vw = Poly.vector_ew(v)

        core_vd = jnp.where(
            uc[..., 3:-2, :] > 0.0,
            ve[..., 2:-3, :],
            vw[..., 3:-2, :],
        )
        pad_vd = [(0, 0)] * (v.ndim - 2) + [(3, 2), (0, 0)]
        vd = jnp.pad(core_vd, pad_vd)

        core_ud = jnp.where(
            vc[..., :, 3:-2] > 0.0,
            un[..., :, 2:-3],
            us[..., :, 3:-2],
        )
        pad_ud = [(0, 0)] * (u.ndim - 2) + [(0, 0), (3, 2)]
        ud = jnp.pad(core_ud, pad_ud)

        return ud, vd

    @staticmethod
    def vector_trans_2d(u, v, a_gct, a_sina):
        """
        Full 2D vector transformation: A-grid → C-grid (with communication) → D-grid (upwind).

        Returns:
            (uct, vct, ub_cx, ub_cy, vb_cx, vb_cy)
        """
        uct, vct = Remap.to_a_grid(u, v, a_gct, a_sina)
        ub_cx, vb_cy = Remap.to_c_grid(uct, vct)
        ub_cx, vb_cy = Communication.boundary_communication(ub_cx, vb_cy)
        ub_cy, vb_cx = Remap.to_d_grid_upwind(u, v, ub_cx, vb_cy)
        return uct, vct, ub_cx, ub_cy, vb_cx, vb_cy
