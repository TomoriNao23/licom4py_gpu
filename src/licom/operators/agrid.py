"""
File: agrid.py
Description: A-grid operators as pure functions.
    Includes vorticity, divergence, and gradient.
    All grid data passed explicitly — no singleton access.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-21
Updated: 2026-04-13

REVISION HISTORY:
    21/09/2025 - Initial implementation
    19/03/2026 - Refactor imports to package-level paths
    19/03/2026 - Wrap functions as AGrid static methods
    07/04/2026 - Replaced zeros_like buffer scatters with native jnp.pad
    13/04/2026 - Converted to explicit parameter passing
"""

# Standard library imports
from typing import Tuple

# Third-party imports
import jax.numpy as jnp

from .poly import Poly


class AGrid:
    """A-grid differential operators (vorticity, divergence, gradient)."""

    @staticmethod
    def vorticity(vv, uu, rda, d_dx, c_dy):
        """Calculate the vorticity on the A-grid."""
        core = rda[..., :-1, :-1] * (
            uu[..., :-1, :-1] * d_dx[..., :-1, :-1]
            + vv[..., 1:, :-1] * c_dy[..., 1:, :-1]
            - uu[..., :-1, 1:] * d_dx[..., :-1, 1:]
            - vv[..., :-1, :-1] * c_dy[..., :-1, :-1]
        )
        pad = [(0, 0)] * (uu.ndim - 2) + [(0, 1), (0, 1)]
        return jnp.pad(core, pad)

    @staticmethod
    def div(uu, vv, rda, c_dy, d_dx):
        """Calculate the divergence on the A-grid."""
        core = rda[..., :-1, :-1] * (
            uu[..., 1:, :-1] * c_dy[..., 1:, :-1]
            - uu[..., :-1, :-1] * c_dy[..., :-1, :-1]
            + vv[..., :-1, 1:] * d_dx[..., :-1, 1:]
            - vv[..., :-1, :-1] * d_dx[..., :-1, :-1]
        )
        pad = [(0, 0)] * (uu.ndim - 2) + [(0, 1), (0, 1)]
        return jnp.pad(core, pad)

    @staticmethod
    def grad(eta, rdx, rdy) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Calculate the gradient on the A-grid."""
        uhalf = Poly.scalar_x(eta)
        vhalf = Poly.scalar_y(eta)

        core_x = rdx[..., 3:-3, 3:-3] * (
            uhalf[..., 4:-2, 3:-3] - uhalf[..., 3:-3, 3:-3]
        )
        core_y = rdy[..., 3:-3, 3:-3] * (
            vhalf[..., 3:-3, 4:-2] - vhalf[..., 3:-3, 3:-3]
        )

        pad = [(0, 0)] * (eta.ndim - 2) + [(3, 3), (3, 3)]
        return jnp.pad(core_x, pad), jnp.pad(core_y, pad)