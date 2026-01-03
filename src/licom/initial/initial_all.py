"""
File: initial_all.py
Description: Main initialization class for LICOM model, setting up namelist,
    MPP configuration, and duogrid for the cubed-sphere mosaic.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2026-01-04

REVISION HISTORY:
    03/09/2025 - Initial implementation of Initial class
    04/01/2026 - Added timer to initialization steps
"""
# Standard library imports
import sys
import os

# Local application imports
from mymodule import Schedule, Timer, get_all_time
from datatype import momentum_data
from operators.agrid import agrid_vorticity, agrid_div, agrid_grad
from readnamelist import Namelist
from backend import FMS_chtholly
from backend.calculation.field import Field
from duogrid.duogrid import Duogrid as Dg
from operators.poly import vector_interpolation_ew, scalar_interpolation_x, scalar_interpolation_y
from operators.remap import to_c_grid, to_d_grid, to_d_grid_upwind, to_a_grid, vector_trans_2d
from momentum.momentum import Momentum
from initial.w92_field import initialize_test_velocity_field

import jax.numpy as jnp
import numpy as np

class Initial:

    namelist: Namelist
    
    def __init__(self):

        with Timer(name = "initial", enabled=True):
            # namelist init
            self.namelist = Namelist.create()

            # backend.FMS_chtholly init
            FMS_chtholly.init(self.namelist)

            # backend.calculation init
            Field.init(FMS_chtholly.mp)

            # duogrid init
            Dg.init(FMS_chtholly.mp)

            # momentum init
            self.momentum = Momentum()

            # schedule init
            Schedule.init(self.namelist, ["barotropic"])

        time_init = get_all_time()
        if FMS_chtholly.mp.pe == 0:
            print(f"Initialization completed successfully in {time_init[0].to_dict()['mean']:.4f} seconds.")

    def __del__(self):
        """
        Destructor: automatically called when the object is about to be destroyed.
        Ensures FMS end is called, similar to how __init__ is used for initialization.
        """
        FMS_chtholly.end()