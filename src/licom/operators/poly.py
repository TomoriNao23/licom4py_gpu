"""
File: poly.py
Description: Polynomial interpolation functions as static methods of the Poly class.
    Uses 3rd order polynomial interpolation only.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-20
Updated: 2026-03-19

REVISION HISTORY:
    20/09/2025 - Initial implementation (vector only)
    21/09/2025 - Added scalar methods
    19/03/2026 - Wrap functions as Poly static methods
"""

# Third-party imports
import functools
import jax
import jax.numpy as jnp

# Standard library imports
from typing import Tuple

# 3rd order polynomial coefficients
Ep23vm = -1.0/20.0
Ep13vm =  9.0/20.0
Ep03vm = 47.0/60.0
Em13vm = -13.0/60.0
Em23vm =  1.0/30.0

M13vm =  37.0/60.0
M23vm =  -2.0/15.0
M33vm =   1.0/60.0


class Poly:
    """3rd-order polynomial interpolation operators."""

    @staticmethod
    def vector_ew(eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Vector polynomial interpolation for East-West direction (3rd order).

        Args:
            eta: input field [..., xsize, ysize]

        Returns:
            (eta_star_east, eta_star_west)
        """
        eta_star_e = jnp.zeros_like(eta).at[..., 2:-2, :].set(
            Em23vm * eta[..., 0:-4, :] +
            Em13vm * eta[..., 1:-3, :] +
            Ep03vm * eta[..., 2:-2, :] +
            Ep13vm * eta[..., 3:-1, :] +
            Ep23vm * eta[..., 4:,   :]
        )
        eta_star_w = jnp.zeros_like(eta).at[..., 2:-2, :].set(
            Ep23vm * eta[..., 0:-4, :] +
            Ep13vm * eta[..., 1:-3, :] +
            Ep03vm * eta[..., 2:-2, :] +
            Em13vm * eta[..., 3:-1, :] +
            Em23vm * eta[..., 4:,   :]
        )
        return eta_star_e, eta_star_w

    @staticmethod
    @functools.partial(jax.jit, static_argnums=())
    def vector_ns(eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Vector polynomial interpolation for North-South direction (3rd order).

        Args:
            eta: input field [..., xsize, ysize]

        Returns:
            (eta_star_north, eta_star_south)
        """
        eta_star_n = jnp.zeros_like(eta).at[..., :, 2:-2].set(
            Em23vm * eta[..., :, 0:-4] +
            Em13vm * eta[..., :, 1:-3] +
            Ep03vm * eta[..., :, 2:-2] +
            Ep13vm * eta[..., :, 3:-1] +
            Ep23vm * eta[..., :, 4:  ]
        )
        eta_star_s = jnp.zeros_like(eta).at[..., :, 2:-2].set(
            Ep23vm * eta[..., :, 0:-4] +
            Ep13vm * eta[..., :, 1:-3] +
            Ep03vm * eta[..., :, 2:-2] +
            Em13vm * eta[..., :, 3:-1] +
            Em23vm * eta[..., :, 4:  ]
        )
        return eta_star_n, eta_star_s

    @staticmethod
    @functools.partial(jax.jit, static_argnums=())
    def vector(eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Vector polynomial interpolation in all four directions (3rd order).

        Returns:
            (eta_e, eta_w, eta_n, eta_s)
        """
        eta_e, eta_w = Poly.vector_ew(eta)
        eta_n, eta_s = Poly.vector_ns(eta)
        return eta_e, eta_w, eta_n, eta_s

    @staticmethod
    @functools.partial(jax.jit, static_argnums=())
    def scalar_x(scal: jnp.ndarray) -> jnp.ndarray:
        """
        Scalar polynomial interpolation in x direction (3rd order).

        Args:
            scal: input scalar field [..., xsize, ysize]

        Returns:
            scal_out: interpolated scalar field
        """
        return jnp.zeros_like(scal).at[..., 3:-2, 3:-2].set(
            M13vm * (scal[..., 2:-3, 3:-2] + scal[..., 3:-2, 3:-2]) +
            M23vm * (scal[..., 1:-4, 3:-2] + scal[..., 4:-1, 3:-2]) +
            M33vm * (scal[..., :-5,  3:-2] + scal[..., 5:,   3:-2])
        )

    @staticmethod
    @functools.partial(jax.jit, static_argnums=())
    def scalar_y(scal: jnp.ndarray) -> jnp.ndarray:
        """
        Scalar polynomial interpolation in y direction (3rd order).

        Args:
            scal: input scalar field [..., xsize, ysize]

        Returns:
            scal_out: interpolated scalar field
        """
        return jnp.zeros_like(scal).at[..., 3:-2, 3:-2].set(
            M13vm * (scal[..., 3:-2, 2:-3] + scal[..., 3:-2, 3:-2]) +
            M23vm * (scal[..., 3:-2, 1:-4] + scal[..., 3:-2, 4:-1]) +
            M33vm * (scal[..., 3:-2, :-5 ] + scal[..., 3:-2, 5:   ])
        )

    @staticmethod
    @functools.partial(jax.jit, static_argnums=())
    def scalar_xy(scal: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Scalar polynomial interpolation in both x and y directions (3rd order).

        Returns:
            (scal_x, scal_y)
        """
        return Poly.scalar_x(scal), Poly.scalar_y(scal)