"""
File: initial_all.py
Description: Main initialization class for LICOM model.
    Refactored to remove Fortran/MPI dependencies, using native JAX.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2026-03-19

REVISION HISTORY:
    03/09/2025 - Initial implementation of Initial class
    04/01/2026 - Added timer to initialization steps
    15/03/2026 - Removed FMS_chtholly, use NPZ + GPU_Mesh + Global2Local
    19/03/2026 - Refactor imports to package-level paths
"""

# Third-party imports
import jax.numpy as jnp

# Local application imports
from licom.duogrid import Dg
from licom.kernel import Global2Local, GPU_Mesh
from licom.momentum import Momentum
from licom.mymodule import Schedule
from licom.readnamelist import Namelist


class Initial:

    namelist: Namelist

    def __init__(self):

        # Namelist
        self.namelist = Namelist.create()

        # GPU Mesh (sets up JAX device mesh, sharding, Global2Local, Communication)
        GPU_Mesh.configure(self.namelist)

        # Duogrid (loads NPZ and distributes via Global2Local)
        Dg.configure(self.namelist, GPU_Mesh)

        # Momentum
        self.momentum = Momentum(self.namelist)

        # Schedule
        Schedule.configure(self.namelist, self.momentum, ["barotropic"])
