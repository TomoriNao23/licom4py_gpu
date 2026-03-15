"""
File: initial_all.py
Description: Main initialization class for LICOM model.
    Refactored to remove Fortran/MPI dependencies, using native JAX.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2026-03-15

REVISION HISTORY:
    03/09/2025 - Initial implementation of Initial class
    04/01/2026 - Added timer to initialization steps
    15/03/2026 - Removed FMS_chtholly, use NPZ + GPU_Mesh + Global2Local
"""

# Local application imports
from mymodule import Schedule, Timer, get_all_time
from readnamelist import Namelist
from mesh.g2l import Global2Local
from duogrid import Dg
from mesh.gpu_mesh import GPU_Mesh
from momentum.momentum import Momentum

# Third-party imports
import jax.numpy as jnp


class Initial:

    namelist: Namelist

    def __init__(self):

        with Timer(name="initial", enabled=True):
            # 1. Namelist
            self.namelist = Namelist.create()

            # 2. GPU Mesh (sets up JAX device mesh, sharding, Global2Local, Communication)
            GPU_Mesh.configure(self.namelist)

            # 3. Global2Local configuration (already done in GPU_Mesh.configure)
            pass

            # 4. Duogrid (loads NPZ and distributes via Global2Local)
            Dg.configure(self.namelist, GPU_Mesh)

            # 5. Momentum
            self.momentum = Momentum(self.namelist)

            # 6. Schedule
            Schedule.configure(self.namelist, ["barotropic"])

        time_init = get_all_time()
        print(f"Initialization completed successfully in {time_init[0].to_dict()['mean']:.4f} seconds.")