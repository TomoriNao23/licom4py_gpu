"""
File: gpu_mesh.py
Description: Main definition and configuration of the GPU mesh and its layout.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-03-14
Updated: 2026-03-14

REVISION HISTORY:
    14/03/2026 - Formatting and header updates
"""
# Third-party imports
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P

# Local application imports
from .g2l import Global2Local
from .communication import Communication


class GPUMesh:
    
    @classmethod
    def configure(cls, namelist):
        cls.ntile = 6
        cls.halo = 3

        cls.nx = namelist.nx
        cls.ny = namelist.ny

        cls.pdev = namelist.pdev
        cls.px = namelist.px
        cls.py = namelist.py

        cls.nx_local = cls.nx // cls.px
        cls.ny_local = cls.ny // cls.py

        assert cls.pdev == len(jax.devices()), f"Number of devices must match the number of tiles: {pdev} != {len(jax.devices())}"

        cls.devices     = np.array(jax.devices()).reshape(cls.pdev, cls.px, cls.py)
        cls.mesh        = Mesh(cls.devices, ('tile', 'x', 'y'))
        cls.sharding_2d = NamedSharding(cls.mesh, P('tile', 'x', 'y'))

        cls.Global2Local = Global2Local.configure(cls.sharding_2d, cls.halo, cls.nx_local, cls.ny_local)
        cls.Communication = Communication.configure(cls.halo, cls.nx_local, cls.ny_local, cls.mesh)
        