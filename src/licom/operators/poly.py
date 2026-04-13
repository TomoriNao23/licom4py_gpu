"""
File: poly.py
Description: Polynomial interpolation functions as static methods of the Poly class.
    Uses 3rd order polynomial interpolation only.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-20
Updated: 2026-04-07

REVISION HISTORY:
    20/09/2025 - Initial implementation (vector only)
    21/09/2025 - Added scalar methods
    19/03/2026 - Wrap functions as Poly static methods
    07/04/2026 - Replaced zeros_like buffer scatters with native jnp.pad
"""

# Standard library imports
import functools
from typing import Tuple

# Third-party imports
import jax
import jax.numpy as jnp

class Poly:
    """3rd-order polynomial interpolation operators."""

    # 3rd order polynomial coefficients
    Ep23vm = -1.0 / 20.0
    Ep13vm = 9.0 / 20.0
    Ep03vm = 47.0 / 60.0
    Em13vm = -13.0 / 60.0
    Em23vm = 1.0 / 30.0

    M13vm = 37.0 / 60.0
    M23vm = -2.0 / 15.0
    M33vm = 1.0 / 60.0

    @staticmethod
    def vector_ew(eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Vector polynomial interpolation for East-West direction (3rd order).

        Args:
            eta: input field [..., xsize, ysize]

        Returns:
            (eta_star_east, eta_star_west)
        """
        core_e = (
            Poly.Em23vm * eta[..., 0:-4, :]
            + Poly.Em13vm * eta[..., 1:-3, :]
            + Poly.Ep03vm * eta[..., 2:-2, :]
            + Poly.Ep13vm * eta[..., 3:-1, :]
            + Poly.Ep23vm * eta[..., 4:, :]
        )
        core_w = (
            Poly.Ep23vm * eta[..., 0:-4, :]
            + Poly.Ep13vm * eta[..., 1:-3, :]
            + Poly.Ep03vm * eta[..., 2:-2, :]
            + Poly.Em13vm * eta[..., 3:-1, :]
            + Poly.Em23vm * eta[..., 4:, :]
        )
        pad = [(0, 0)] * (eta.ndim - 2) + [(2, 2), (0, 0)]
        return jnp.pad(core_e, pad), jnp.pad(core_w, pad)

    @staticmethod
    def vector_ns(eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Vector polynomial interpolation for North-South direction (3rd order).

        Args:
            eta: input field [..., xsize, ysize]

        Returns:
            (eta_star_north, eta_star_south)
        """
        core_n = (
            Poly.Em23vm * eta[..., :, 0:-4]
            + Poly.Em13vm * eta[..., :, 1:-3]
            + Poly.Ep03vm * eta[..., :, 2:-2]
            + Poly.Ep13vm * eta[..., :, 3:-1]
            + Poly.Ep23vm * eta[..., :, 4:]
        )
        core_s = (
            Poly.Ep23vm * eta[..., :, 0:-4]
            + Poly.Ep13vm * eta[..., :, 1:-3]
            + Poly.Ep03vm * eta[..., :, 2:-2]
            + Poly.Em13vm * eta[..., :, 3:-1]
            + Poly.Em23vm * eta[..., :, 4:]
        )
        pad = [(0, 0)] * (eta.ndim - 2) + [(0, 0), (2, 2)]
        return jnp.pad(core_n, pad), jnp.pad(core_s, pad)

    @staticmethod
    def vector(
        eta: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Vector polynomial interpolation in all four directions (3rd order).

        Returns:
            (eta_e, eta_w, eta_n, eta_s)
        """
        eta_e, eta_w = Poly.vector_ew(eta)
        eta_n, eta_s = Poly.vector_ns(eta)
        return eta_e, eta_w, eta_n, eta_s

    @staticmethod
    def scalar_x(scal: jnp.ndarray) -> jnp.ndarray:
        """
        Scalar polynomial interpolation in x direction (3rd order).

        Args:
            scal: input scalar field [..., xsize, ysize]

        Returns:
            scal_out: interpolated scalar field
        """
        core = (
            Poly.M13vm * (scal[..., 2:-3, 3:-2] + scal[..., 3:-2, 3:-2])
            + Poly.M23vm * (scal[..., 1:-4, 3:-2] + scal[..., 4:-1, 3:-2])
            + Poly.M33vm * (scal[..., :-5, 3:-2] + scal[..., 5:, 3:-2])
        )
        pad = [(0, 0)] * (scal.ndim - 2) + [(3, 2), (3, 2)]
        return jnp.pad(core, pad)

    @staticmethod
    def scalar_y(scal: jnp.ndarray) -> jnp.ndarray:
        """
        Scalar polynomial interpolation in y direction (3rd order).

        Args:
            scal: input scalar field [..., xsize, ysize]

        Returns:
            scal_out: interpolated scalar field
        """
        core = (
            Poly.M13vm * (scal[..., 3:-2, 2:-3] + scal[..., 3:-2, 3:-2])
            + Poly.M23vm * (scal[..., 3:-2, 1:-4] + scal[..., 3:-2, 4:-1])
            + Poly.M33vm * (scal[..., 3:-2, :-5] + scal[..., 3:-2, 5:])
        )
        pad = [(0, 0)] * (scal.ndim - 2) + [(3, 2), (3, 2)]
        return jnp.pad(core, pad)

    @staticmethod
    def scalar_xy(scal: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Scalar polynomial interpolation in both x and y directions (3rd order).

        Returns:
            (scal_x, scal_y)
        """
        return Poly.scalar_x(scal), Poly.scalar_y(scal)