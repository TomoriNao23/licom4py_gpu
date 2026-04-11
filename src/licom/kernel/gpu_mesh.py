"""
File: gpu_mesh.py
Description: Main definition and configuration of the GPU mesh and its layout.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-03-14
Updated: 2026-03-15
"""

# Third-party imports
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import SingleDeviceSharding

from .communication import Communication
from .g2l import Global2Local


class GPU_Mesh:

    @classmethod
    def configure(cls, namelist):
        cls.ntile = 6
        cls.halo = 3

        cls.nx = namelist.nx
        cls.ny = namelist.ny
        cls.npz = namelist.npz

        cls.pdev = namelist.pdev
        cls.px = namelist.px
        cls.py = namelist.py

        cls.nx_local = cls.nx // cls.px
        cls.ny_local = cls.ny // cls.py

        # JAX devices
        devices = jax.devices()

        # As per user convention: always construct the mesh as (pdev, px, py),
        # eliminating the need for extra fallback branches. pdev, px, py are controlled by the namelist,
        # so even a single-device setup maps cleanly to (1, 1, 1).
        n_mesh = cls.pdev * cls.px * cls.py
        if len(devices) < n_mesh:
            raise ValueError(
                f"Not enough JAX devices: need pdev*px*py={n_mesh}, "
                f"but got {len(devices)}. Please adjust namelist.pdev/px/py or devices."
            )

        used = np.array(devices[:n_mesh]).reshape(cls.pdev, cls.px, cls.py)
        cls.devices = used
        cls.mesh = Mesh(cls.devices, ("tile", "x", "y"))
        # cls.sharding_2d = NamedSharding(cls.mesh, P('tile', 'x', 'y'))

        # Global2Local and Communication now use this sharding
        Global2Local.configure(
            cls.mesh,
            cls.halo,
            cls.nx_local,
            cls.ny_local,
            cls.nx,
            cls.ny,
            cls.npz,
            cls.ntile,
        )
        Communication.configure(cls.halo, cls.nx_local, cls.ny_local, cls.mesh)
