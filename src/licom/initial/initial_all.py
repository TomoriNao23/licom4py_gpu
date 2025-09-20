"""
File: initial_all.py
Description: Main initialization class for LICOM model, setting up namelist,
    MPP configuration, and duogrid for the cubed-sphere mosaic.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-16
"""
# Standard library imports
import sys
import os


# Local application imports
from readnamelist import Namelist
from backend import FMS_chtholly
from backend.calculation.field import Field
from duogrid.duogrid import Duogrid as Dg
from operators.poly import vector_interpolation_ew
from operators.remap import to_c_grid, to_d_grid, to_d_grid_upwind

import jax.numpy as jnp
import numpy as np

class Initial:

    namelist: Namelist
    
    def __init__(self):

        # namelist init
        self.namelist = Namelist.create()

        # backend.FMS_chtholly init
        FMS_chtholly.init(self.namelist)

        # backend.calculation init
        Field.init(FMS_chtholly.mp)

        # duogrid init
        Dg.init(FMS_chtholly.mp)

        u = Field.new('2d')
        v = Field.new('2d')
        
        # Vectorized initialization using JAX broadcasting
        i_indices = jnp.arange(Dg.mp.xsize) + Dg.mp.isd  # [0, 1, 2, ..., xsize-1] + isd
        j_indices = jnp.arange(Dg.mp.ysize) + Dg.mp.jsd  # [0, 1, 2, ..., ysize-1] + jsd
        
        # Create meshgrid for broadcasting
        i_grid, j_grid = jnp.meshgrid(i_indices, j_indices, indexing='ij')
        
        # Vectorized computation
        u_values = Dg.mp.tile * 10000 + i_grid * 100 + j_grid
        v_values = u_values * 1.1
        
        # Set the values using vectorized assignment
        u = u.at[:].set(u_values)
        v = v.at[:].set(v_values)
        uc, vc = to_c_grid(u, v)
        ud, vd = to_d_grid(u, v)
        ud_upwind, vd_upwind = to_d_grid_upwind(u, v, uc, vc)
        print("cpu", Dg.mp.pe, "sum", f"{jnp.sum(uc):.8f}") if (Dg.mp.pe == 6) else None
        print("cpu", Dg.mp.pe, "sum", f"{jnp.sum(vc):.8f}") if (Dg.mp.pe == 6) else None
        print("cpu", Dg.mp.pe, "sum", f"{jnp.sum(ud):.8f}") if (Dg.mp.pe == 6) else None
        print("cpu", Dg.mp.pe, "sum", f"{jnp.sum(vd):.8f}") if (Dg.mp.pe == 6) else None
        print("cpu", Dg.mp.pe, "sum", f"{jnp.sum(ud_upwind):.8f}") if (Dg.mp.pe == 6) else None
        print("cpu", Dg.mp.pe, "sum", f"{jnp.sum(vd_upwind):.8f}") if (Dg.mp.pe == 6) else None
    def __del__(self):
        """
        Destructor: automatically called when the object is about to be destroyed.
        Ensures FMS end is called, similar to how __init__ is used for initialization.
        """
        FMS_chtholly.end()


        # u = Field.new('2d')
        # v = Field.new('2d')
        # for i in range(Dg.mp.xsize):
        #     for j in range(Dg.mp.ysize):
        #         #a[i,j] = self.mp.tile*100 + (i-2) + (j-2)*0.01
        #         u=u.at[i,j].set(Dg.mp.tile*10000 + (i+Dg.mp.isd)*100 + (j+Dg.mp.jsd))
        #         v=v.at[i,j].set((Dg.mp.tile*10000 + (i+Dg.mp.isd)*100 + (j+Dg.mp.jsd))*1.1)
        # u, v = FMS_chtholly.ext_vector(u, v)
        # print("cpu", Dg.mp.pe, "sum", f"{jnp.sum(u):.8f}") if (Dg.mp.pe == 6) else None
        # print("cpu", Dg.mp.pe, "sum", f"{jnp.sum(v):.8f}") if (Dg.mp.pe == 6) else None
        # u, v = FMS_chtholly.communication2d(u, v)
        # print("cpu", Dg.mp.pe, "sum", f"{jnp.sum(u):.8f}") if (Dg.mp.pe == 6) else None
        # print("cpu", Dg.mp.pe, "sum", f"{jnp.sum(v):.8f}") if (Dg.mp.pe == 6) else None
        # if Dg.mp.pe == 6:
        #     for j in range(Dg.mp.ysize):
        #         for i in range(Dg.mp.xsize):
        #             print("i=",Dg.mp.isd+i, "j=",Dg.mp.jsd+j, u[i,j])
        # if Dg.mp.pe == 6:
        #     for j in range(Dg.mp.ysize):
        #         for i in range(Dg.mp.xsize):
        #             print("i=",Dg.mp.isd+i, "j=",Dg.mp.jsd+j, Dg.inner[i,j])


