"""
File: agrid.py
Description: A-grid operators as static methods of the AGrid class.
    Includes vorticity, divergence, and gradient.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-21
Updated: 2026-04-07

REVISION HISTORY:
    21/09/2025 - Initial implementation
    19/03/2026 - Refactor imports to package-level paths
    19/03/2026 - Wrap functions as AGrid static methods
    07/04/2026 - Replaced zeros_like buffer scatters with native jnp.pad
"""

# Third-party imports
import jax
import jax.numpy as jnp
import functools

# Standard library imports
from typing import Tuple

# Local application imports
from licom.duogrid import Dg
from .poly import Poly


class AGrid:
    """A-grid differential operators (vorticity, divergence, gradient)."""

    @staticmethod
    def vorticity(vv: jnp.ndarray, uu: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the vorticity on the A-grid.

        Args:
            vv: v velocity component [xsize, ysize]
            uu: u velocity component [xsize, ysize]

        Formula:
            vort(i,j) = rda(i,j) * (
                uu(i,j) * d_dx(i,j) +
                vv(i+1,j) * c_dy(i+1,j) -
                uu(i,j+1) * d_dx(i,j+1) +
                vv(i,j) * c_dy(i,j)
            )

        Returns:
            vort: vorticity [xsize, ysize]
        """
        core = Dg.rda[..., :-1, :-1] * (
            uu[..., :-1, :-1] * Dg.d_dx[..., :-1, :-1] +
            vv[..., 1:, :-1] * Dg.c_dy[..., 1:, :-1] -        # vv(i+1,j) * c_dy(i+1,j)
            uu[..., :-1, 1:] * Dg.d_dx[..., :-1, 1:] -        # uu(i,j+1) * d_dx(i,j+1)
            vv[..., :-1, :-1] * Dg.c_dy[..., :-1, :-1]          # vv(i,j) * c_dy(i,j)
        )
        pad = [(0, 0)] * (uu.ndim - 2) + [(0, 1), (0, 1)]
        return jnp.pad(core, pad)

    @staticmethod
    def div(uu: jnp.ndarray, vv: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the divergence on the A-grid.

        Args:
            uu: u velocity component [xsize, ysize]
            vv: v velocity component [xsize, ysize]

        Returns:
            div: divergence [xsize, ysize]
        """
        core = Dg.rda[..., :-1, :-1] * (
            uu[..., 1:, :-1] * Dg.c_dy[..., 1:, :-1] -        # UX(i+1,j) * c_dy(i+1,j)
            uu[..., :-1, :-1] * Dg.c_dy[..., :-1, :-1] +        # UX(i,j) * c_dy(i,j)
            vv[..., :-1, 1:] * Dg.d_dx[..., :-1, 1:] -        # UY(i,j+1) * d_dx(i,j+1)
            vv[..., :-1, :-1] * Dg.d_dx[..., :-1, :-1]          # UY(i,j) * d_dx(i,j)
        )
        pad = [(0, 0)] * (uu.ndim - 2) + [(0, 1), (0, 1)]
        return jnp.pad(core, pad)

    @staticmethod
    def grad(eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculate the gradient on the A-grid.

        Args:
            eta: scalar field [xsize, ysize]

        Returns:
            gradx: gradient in x direction [xsize, ysize]
            grady: gradient in y direction [xsize, ysize]
        """
        uhalf = Poly.scalar_x(eta)
        vhalf = Poly.scalar_y(eta)

        core_x = Dg.rdx[..., 3:-3, 3:-3] * (uhalf[..., 4:-2, 3:-3] - uhalf[..., 3:-3, 3:-3])
        core_y = Dg.rdy[..., 3:-3, 3:-3] * (vhalf[..., 3:-3, 4:-2] - vhalf[..., 3:-3, 3:-3])
        
        pad = [(0, 0)] * (eta.ndim - 2) + [(3, 3), (3, 3)]
        return jnp.pad(core_x, pad), jnp.pad(core_y, pad)