"""
File: w92_field.py
Description: Velocity field transformation functions for converting between
    spherical (lat-lon) coordinates and cubed-sphere grid coordinates.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-22
Updated: 2026-05-14

REVISION HISTORY:
    22/09/2025 - Initial implementation of velocity transformation
    19/03/2026 - Consolidate typing imports; refactor imports to package-level paths
    14/05/2026 - Move initialization to CPU Host
"""

# Standard library imports
import functools
import os
from typing import Any, Optional, Tuple

# Third-party imports
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

# Local application imports
from licom.duogrid import Dg
from licom.kernel import Communication, GPU_Mesh
from licom.kernel.g2l import Global2Local
from licom.operators import Remap


def initialize_test_velocity_field(momentum=None, test_case: str = "w92case2") -> None:
    """
    Initialize test velocity fields by computing globally on the host,
    and then distributing to each card.
    Resolves analytical steady-state conditions and synchronizes corner padding globally.
    """
    if test_case == "w92case2":
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        npz_path = os.path.join(project_root, "field", f"duogrid_C{Global2Local.nx}.npz")
        
        # Load the raw (but padded) data from the NPZ file
        data = np.load(npz_path)
        a_pt_padded = jnp.array(data["a_pt"], dtype=Global2Local.dtype)
        a_l2c_padded = jnp.array(data["a_l2c"], dtype=Global2Local.dtype)
        
        lon = a_pt_padded[..., 0]
        lat = a_pt_padded[..., 1]
        
        ubar = 1.0
        alpha = 0.0
        radius = 6.371e6
        omega = 7.292e-5
        grav = 9.80
        
        u_rll_lon = ubar * (
            jnp.cos(alpha) * jnp.cos(lat)
            + jnp.sin(alpha) * jnp.cos(lon) * jnp.sin(lat)
        )
        u_rll_lat = -ubar * jnp.sin(alpha) * jnp.sin(lon)
        
        u_cubed = (
            a_l2c_padded[..., 0, 0] * u_rll_lon + a_l2c_padded[..., 0, 1] * u_rll_lat
        )
        v_cubed = (
            a_l2c_padded[..., 1, 0] * u_rll_lon + a_l2c_padded[..., 1, 1] * u_rll_lat
        )
        
        h0_int = (
            -(radius * omega * ubar + ubar**2 / 2.0)
            * (
                (
                    -jnp.cos(lon) * jnp.cos(lat) * jnp.sin(alpha)
                    + jnp.sin(lat) * jnp.cos(alpha)
                )
                ** 2
                - 1.0 / 3.0
            )
            / grav
        )
        
        # Distribute the global pre-padded arrays to each GPU directly
        momentum.ub = Global2Local.distribute_pre_padded(u_cubed)
        momentum.vb = Global2Local.distribute_pre_padded(v_cubed)
        momentum.h0 = Global2Local.distribute_pre_padded(h0_int)

    # ubp, vbp, h0p
    momentum.ubp = momentum.ub
    momentum.vbp = momentum.vb
    momentum.h0p = momentum.h0
