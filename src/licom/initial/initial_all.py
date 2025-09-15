"""
File: initial_all.py
Description: Main initialization class for LICOM model, setting up namelist,
    MPP configuration, and duogrid for the cubed-sphere mosaic.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-11
"""
# Standard library imports
import sys
import os


# Local application imports
from readnamelist import Namelist
from backend.cube_grid.use_mpp import FMS_chtholly
# test
import jax.numpy as jnp
import numpy as np

class Initial:

    namelist: Namelist

    def __init__(self):

        # namelist init
        self.namelist = Namelist.create()

        FMS_chtholly.init()


    def __del__(self):
        """
        Destructor: automatically called when the object is about to be destroyed.
        Ensures FMS end is called, similar to how __init__ is used for initialization.
        """
        FMS_chtholly.end()

